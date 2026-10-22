from typing import Tuple
import timm
import torch
import torch.nn as nn

def create_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Create a timm vision model with ImageNet pretraining adapted for paddy disease classification."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all layers except the classification head (Stage 1)."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head
    if hasattr(model, "get_classifier"):
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise AttributeError(f"Could not locate classifier head in {model_name}")

def unfreeze_last_blocks(model: nn.Module, model_name: str) -> None:
    """Unfreeze the top blocks and conv head for fine-tuning (Stage 2)."""
    # Always keep classifier trainable
    if hasattr(model, "get_classifier"):
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Unfreeze conv_head / bn2 if present
    for attr in ["conv_head", "bn2", "head"]:
        if hasattr(model, attr):
            for param in getattr(model, attr).parameters():
                param.requires_grad = True

    # Unfreeze last blocks in feature extractor
    if hasattr(model, "blocks"):
        # For mobilenetv3 and efficientnet, blocks is a Sequential or ModuleList
        for block in model.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

def get_trainable_param_counts(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

if __name__ == "__main__":
    for m in ["mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_b0"]:
        net = create_model(m)
        freeze_backbone(net, m)
        tr1, tot = get_trainable_param_counts(net)
        unfreeze_last_blocks(net, m)
        tr2, tot = get_trainable_param_counts(net)
        print(f"{m:22s} | Total: {tot:,} | Stage 1 Trainable: {tr1:,} | Stage 2 Trainable: {tr2:,}")
