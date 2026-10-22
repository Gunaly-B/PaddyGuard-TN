import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.data.dataset import get_transforms
from src.explainability.gradcam import compute_gradcam_and_attention
from src.models.model_factory import create_model
from src.rag.advisory import GroundedAdvisoryRAG
from src.severity.analyzer import segment_leaf_and_lesions, create_severity_overlay

def compute_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Compute negative log-sum-exp energy score for OOD detection.
    In-distribution samples exhibit lower (more negative) energy, OOD exhibit higher energy."""
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return float(energy.item())

class PaddyPredictor:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model_name: str = "mobilenetv3_small_100",
        splits_path: Path = Path("data/processed/splits.json"),
        device: torch.device = torch.device("cpu"),
        ood_msp_threshold: float = 0.65,
        ood_energy_threshold: float = -3.0
    ):
        self.device = device
        self.ood_msp_threshold = ood_msp_threshold
        self.ood_energy_threshold = ood_energy_threshold

        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        self.class_names = splits_data["metadata"]["class_names"]

        # Default to best available checkpoint if none provided
        if checkpoint_path is None:
            ckpt_files = list(Path("results/checkpoints").glob(f"{model_name}/*best.pt"))
            if ckpt_files:
                checkpoint_path = sorted(ckpt_files)[0]
            else:
                checkpoint_path = None

        print(f"[Predictor] Initializing {model_name}...")
        self.model_name = model_name
        self.model = create_model(model_name, num_classes=len(self.class_names), pretrained=False)
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[Predictor] Loading weights from {checkpoint_path}...")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print("[Predictor Warning] No checkpoint loaded; running in initialized state.")

        self.model.eval()
        self.model.to(device)
        self.transform = get_transforms(224, is_train=False)

        print("[Predictor] Initializing Grounded Advisory RAG...")
        self.rag = GroundedAdvisoryRAG()

    def predict_image(self, image_path: Path, output_dir: Optional[Path] = None,
                      include_visuals: bool = False) -> Dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        t0 = time.perf_counter()
        pil_img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            energy = compute_energy_score(logits[0])

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        # Top predictions
        top_indices = np.argsort(probs)[::-1]
        top3 = [{
            "rank": r + 1,
            "class_name": self.class_names[idx],
            "probability": round(float(probs[idx]), 4),
            "percentage": round(float(probs[idx]) * 100.0, 2)
        } for r, idx in enumerate(top_indices[:3])]

        pred_idx = top_indices[0]
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        top_two_margin = float(probs[top_indices[0]] - probs[top_indices[1]]) if len(top_indices) > 1 else 0.0

        # OOD Detection
        is_ood_msp = bool(confidence < self.ood_msp_threshold)
        is_ood_energy = bool(energy > self.ood_energy_threshold)
        is_ood = bool(is_ood_msp or is_ood_energy)
        ood_status = "OOD_SUSPECTED" if is_ood else "IN_DISTRIBUTION"

        # Heuristic Severity Grading (Phase 6)
        img_bgr = cv2.imread(str(image_path))
        leaf_mask, lesion_mask, sev_pct, ses_score, ses_band = segment_leaf_and_lesions(img_bgr)

        # Explainability Grad-CAM (Phase 5)
        gradcam_res = compute_gradcam_and_attention(
            model=self.model,
            model_name=self.model_name,
            image_path=image_path,
            target_class_idx=pred_idx,
            class_names=self.class_names
        )

        # Treatment Advisory (Phase 8). All safety gates apply to every image.
        advisory_reliable = (not is_ood and confidence >= 0.70 and top_two_margin >= 0.15)
        effective_conf = confidence if advisory_reliable else 0.0
        advisory_res = self.rag.get_advisory(
            predicted_disease=pred_class,
            confidence=effective_conf,
            severity_band=ses_band,
            severity_pct=sev_pct
        )

        report = {
            "image_path": str(image_path),
            "filename": image_path.name,
            "inference_time_ms": round(inference_time_ms, 2),
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "top3_predictions": top3,
            "prediction_safety": {
                "top_two_margin": round(top_two_margin, 4),
                "confidence_floor": 0.70,
                "margin_floor": 0.15,
                "advisory_reliable": advisory_reliable
            },
            "ood_detection": {
                "status": ood_status,
                "is_ood": is_ood,
                "max_softmax": round(confidence, 4),
                "energy_score": round(energy, 4),
                "msp_threshold": self.ood_msp_threshold,
                "energy_threshold": self.ood_energy_threshold
            },
            "severity_assessment": {
                "lesion_percentage": sev_pct,
                "irri_ses_score": ses_score,
                "irri_ses_band": ses_band,
                "caveat": "heuristic, not validated against expert scoring"
            },
            "attention_distribution": {
                "lesion_attention_pct": gradcam_res["attention_lesion_pct"],
                "background_attention_pct": gradcam_res["attention_bg_pct"],
                "concentration_ratio": gradcam_res["concentration_ratio"]
            },
            "advisory": advisory_res
        }

        # Print console summary
        self._print_console_report(report)

        # Save artifacts if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # JSON Report
            json_path = output_dir / f"{image_path.stem}_report.json"
            # Remove raw ndarray fields before dump
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            # Composite Visual PNG
            fig_path = output_dir / f"{image_path.stem}_prediction_visual.png"
            self._save_visual_panel(image_path, report, gradcam_res, fig_path)
            report["saved_artifacts"] = {
                "json_report": str(json_path),
                "visual_png": str(fig_path)
            }

        if include_visuals:
            report["_visuals"] = {
                "cam_overlay": gradcam_res["cam_overlay"],
                "leaf_mask": leaf_mask,
                "lesion_mask": lesion_mask,
            }

        return report

    def _print_console_report(self, r: Dict):
        print("\n" + "="*70)
        print(f" PADDY DISEASE DIAGNOSIS REPORT: {r['filename']}")
        print("="*70)
        print(f"  Inference Latency:   {r['inference_time_ms']} ms")
        print(f"  Predicted Disease:   {r['predicted_class']} (Confidence: {r['confidence']*100:.2f}%)")
        print(f"  Top-3 Probabilities:")
        for t in r["top3_predictions"]:
            print(f"     {t['rank']}. {t['class_name']:22s} : {t['percentage']:.2f}%")

        print(f"\n  [OOD Assessment]      Status: {r['ood_detection']['status']}")
        print(f"     Max Softmax:      {r['ood_detection']['max_softmax']:.4f} (threshold: {self.ood_msp_threshold})")
        print(f"     Energy Score:     {r['ood_detection']['energy_score']:.4f} (threshold: {self.ood_energy_threshold})")
        if r["ood_detection"]["is_ood"]:
            print("     >>> WARNING: Suspected Out-of-Distribution or Non-Rice Input! <<<")

        sev = r["severity_assessment"]
        print(f"\n  [Severity Grading]   IRRI SES: Score {sev['irri_ses_score']} ({sev['irri_ses_band']})")
        print(f"     Lesion Area:      {sev['lesion_percentage']}%")
        print(f"     Caveat:           {sev['caveat']}")

        attn = r["attention_distribution"]
        print(f"\n  [Explainability]     Lesion Attn: {attn['lesion_attention_pct']}% | BG Attn: {attn['background_attention_pct']}%")
        print(f"     Attn Density:     {attn['concentration_ratio']}x lesion vs background")

        print(f"\n  [Advisory Status]    {r['advisory']['status']}")
        if r['advisory']['status'] == "SUCCESS":
            print(f"     Citations:        {', '.join(r['advisory']['citations'])}")
        else:
            print(f"     Message:          {r['advisory']['message']}")
        print("="*70 + "\n")

    def _save_visual_panel(self, image_path: Path, report: Dict, cam_res: Dict, output_path: Path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Original
        orig = Image.open(image_path)
        axes[0].imshow(orig)
        axes[0].set_title(f"Input: {image_path.name}\nPred: {report['predicted_class']} ({report['confidence']*100:.1f}%)")
        axes[0].axis("off")

        # 2. Severity Segmentation
        axes[1].imshow(cam_res["leaf_mask"], cmap="gray", alpha=0.3)
        axes[1].imshow(cam_res["lesion_mask"], cmap="Reds", alpha=0.7)
        axes[1].set_title(f"Heuristic Severity Mask\nSES: {report['severity_assessment']['irri_ses_band']} ({report['severity_assessment']['lesion_percentage']}%)")
        axes[1].axis("off")

        # 3. Grad-CAM Overlay
        axes[2].imshow(cam_res["cam_overlay"])
        axes[2].set_title(f"Grad-CAM Attention Overlay\nLesion: {report['attention_distribution']['lesion_attention_pct']}% | BG: {report['attention_distribution']['background_attention_pct']}%")
        axes[2].axis("off")

        ood_text = " [WARNING: OOD INPUT DETECTED]" if report["ood_detection"]["is_ood"] else ""
        plt.suptitle(
            f"Diagnosis: {report['predicted_class']} | Latency: {report['inference_time_ms']}ms{ood_text}\n"
            f"Severity: IRRI SES {report['severity_assessment']['irri_ses_score']} ({report['severity_assessment']['caveat']})",
            fontsize=11, y=0.98
        )
        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Paddy Disease Diagnosis & Advisory Inference Pipeline")
    parser.add_argument("image_path", type=str, nargs="?", default=None, help="Path to single image")
    parser.add_argument("--folder", type=str, default=None, help="Batch mode: folder containing images")
    parser.add_argument("--output_dir", type=str, default="results/predictions", help="Directory to save JSON & PNG reports")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--model", type=str, default="mobilenetv3_small_100", help="Model architecture")
    args = parser.parse_args()

    predictor = PaddyPredictor(checkpoint_path=args.checkpoint, model_name=args.model)

    if args.folder:
        folder = Path(args.folder)
        img_files = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.png"))
        print(f"[Batch Mode] Found {len(img_files)} images in {folder}...")
        for img_p in img_files:
            predictor.predict_image(img_p, output_dir=Path(args.output_dir))
    elif args.image_path:
        predictor.predict_image(Path(args.image_path), output_dir=Path(args.output_dir))
    else:
        print("Please provide an image_path or --folder. Use --help for usage details.")

if __name__ == "__main__":
    main()
