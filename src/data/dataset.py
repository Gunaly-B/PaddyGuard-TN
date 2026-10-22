from pathlib import Path
from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T

class RandomBackgroundBlur:
    """Apply occasional whole-image blur as an appearance augmentation."""
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            radius = np.random.uniform(1.0, 3.0)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class LeafForegroundCrop:
    """Crop around the largest saturated plant-like component when detectable.

    If no sufficiently large component is found, leave the image unchanged.
    This is an opt-in preprocessing ablation; it is not an expert leaf mask.
    """
    def __init__(self, padding_fraction: float = 0.04, min_area_fraction: float = 0.04):
        self.padding_fraction = padding_fraction
        self.min_area_fraction = min_area_fraction

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] > 22) & (hsv[:, :, 2] > 25) & (hsv[:, :, 2] < 250)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return img
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, area = stats[component]
        if area < rgb.shape[0] * rgb.shape[1] * self.min_area_fraction:
            return img
        pad = int(round(max(width, height) * self.padding_fraction))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(rgb.shape[1], x + width + pad), min(rgb.shape[0], y + height + pad)
        return img.crop((x0, y0, x1, y1)) if x1 > x0 and y1 > y0 else img


def get_transforms(image_size: int = 224, is_train: bool = True,
                   background_mode: str = "standard"):
    if background_mode not in {"standard", "leaf_crop"}:
        raise ValueError("background_mode must be 'standard' or 'leaf_crop'")
    foreground_crop = [LeafForegroundCrop()] if background_mode == "leaf_crop" else []
    if is_train:
        return T.Compose([
            *foreground_crop,
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            RandomBackgroundBlur(p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            *foreground_crop,
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class PaddyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = Path(row["rel_path"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = T.ToTensor()(img)

        label = self.class_to_idx[row["class_name"]]
        return tensor_img, label, str(img_path)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean_manifest.csv")
    ds = PaddyDataset(df, transform=get_transforms(224, is_train=True))
    x, y, p = ds[0]
    print(f"[Dataset] Verified PaddyDataset: shape={x.shape}, label={y}, class={ds.idx_to_class[y]}, path={p}")
