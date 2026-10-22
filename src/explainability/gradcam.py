from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.dataset import get_transforms
from src.models.model_factory import create_model
from src.severity.analyzer import segment_leaf_and_lesions

def get_target_layer(model: nn.Module, model_name: str) -> List[nn.Module]:
    """Find the optimal convolutional layer for Grad-CAM."""
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        return [model.blocks[-1]]
    elif hasattr(model, "conv_head"):
        return [model.conv_head]
    elif hasattr(model, "features"):
        return [model.features[-1]]
    else:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                return [module]
        raise ValueError(f"Could not automatically locate target conv layer for {model_name}")

def compute_gradcam_and_attention(
    model: nn.Module,
    model_name: str,
    image_path: Path,
    target_class_idx: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """Compute Grad-CAM heatmap and quantify lesion vs background attention ratio."""
    device = torch.device("cpu")
    model.eval()
    model.to(device)

    # Ensure gradients are tracked for CAM backward pass
    for param in model.parameters():
        param.requires_grad = True

    # 1. Load image and preprocess
    pil_img = Image.open(image_path).convert("RGB")
    w_orig, h_orig = pil_img.size

    # Preprocess for model
    transform = get_transforms(224, is_train=False)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    if target_class_idx is None:
        target_class_idx = pred_idx

    # 2. Grad-CAM execution
    target_layers = get_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0] # (224, 224), float [0, 1]

    # Resize CAM to original image size for accurate overlay against segmentation masks
    cam_resized = cv2.resize(grayscale_cam, (w_orig, h_orig))

    # 3. Extract heuristic segmentation masks (Phase 6)
    img_bgr = cv2.imread(str(image_path))
    leaf_mask, lesion_mask, sev_pct, ses_score, ses_band = segment_leaf_and_lesions(img_bgr)

    # Masks are uint8 [0, 255]
    leaf_bool = (leaf_mask > 0)
    lesion_bool = (lesion_mask > 0)
    bg_bool = ~leaf_bool
    healthy_leaf_bool = leaf_bool & (~lesion_bool)

    # 4. Quantify Attention Distribution
    total_cam = float(np.sum(cam_resized)) + 1e-8
    cam_lesion = float(np.sum(cam_resized[lesion_bool]))
    cam_healthy = float(np.sum(cam_resized[healthy_leaf_bool]))
    cam_bg = float(np.sum(cam_resized[bg_bool]))

    lesion_ratio = (cam_lesion / total_cam) * 100.0
    healthy_ratio = (cam_healthy / total_cam) * 100.0
    bg_ratio = (cam_bg / total_cam) * 100.0

    # Area proportions
    total_pixels = float(w_orig * h_orig)
    lesion_area_prop = float(np.sum(lesion_bool)) / total_pixels + 1e-8
    bg_area_prop = float(np.sum(bg_bool)) / total_pixels + 1e-8

    # Density = attention_prop / area_prop
    lesion_density = (cam_lesion / total_cam) / lesion_area_prop
    bg_density = (cam_bg / total_cam) / bg_area_prop
    concentration_ratio = lesion_density / (bg_density + 1e-8)

    # Normalize image for visualization
    rgb_norm = np.array(pil_img, dtype=np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_norm, cam_resized, use_rgb=True)

    pred_label = class_names[pred_idx] if class_names else str(pred_idx)
    target_label = class_names[target_class_idx] if class_names else str(target_class_idx)

    return {
        "image_path": str(image_path),
        "filename": image_path.name,
        "pred_class": pred_label,
        "pred_prob": float(probs[pred_idx]),
        "all_probs": probs.tolist(),
        "target_class": target_label,
        "cam_resized": cam_resized,
        "cam_overlay": cam_overlay,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
        "severity_pct": sev_pct,
        "ses_band": ses_band,
        "attention_lesion_pct": round(lesion_ratio, 2),
        "attention_healthy_pct": round(healthy_ratio, 2),
        "attention_bg_pct": round(bg_ratio, 2),
        "lesion_attention_density": round(lesion_density, 3),
        "bg_attention_density": round(bg_density, 3),
        "concentration_ratio": round(concentration_ratio, 2)
    }

def plot_gradcam_explanation(exp: Dict, output_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # 1. Original
    orig_img = Image.open(exp["image_path"])
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original\nPred: {exp['pred_class']} ({exp['pred_prob']*100:.1f}%)")
    axes[0].axis("off")

    # 2. Heuristic Lesion Segmentation
    axes[1].imshow(exp["leaf_mask"], cmap="gray", alpha=0.4)
    axes[1].imshow(exp["lesion_mask"], cmap="Reds", alpha=0.7)
    axes[1].set_title(f"Heuristic Mask (Phase 6)\nSeverity: {exp['severity_pct']}% ({exp['ses_band']})")
    axes[1].axis("off")

    # 3. Grad-CAM Heatmap
    im_cam = axes[2].imshow(exp["cam_resized"], cmap="jet")
    axes[2].set_title("Grad-CAM Heatmap\n(Activation Intensity)")
    axes[2].axis("off")
    fig.colorbar(im_cam, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Grad-CAM Overlay with Metrics
    axes[3].imshow(exp["cam_overlay"])
    axes[3].set_title(f"Overlay\nLesion Attn: {exp['attention_lesion_pct']}% | BG Attn: {exp['attention_bg_pct']}%")
    axes[3].axis("off")

    plt.suptitle(
        f"Explainability Analysis: {exp['filename']} | Density Ratio (Lesion/BG): {exp['concentration_ratio']:.2f}x\n"
        f"[Caveat: Heuristic lesion mask is unvalidated; background attention indicates potential shortcut reliance]",
        fontsize=11, y=0.98
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
