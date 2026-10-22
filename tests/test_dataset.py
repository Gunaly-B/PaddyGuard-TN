import numpy as np
from PIL import Image

from src.data.dataset import LeafForegroundCrop, get_transforms


def test_leaf_foreground_crop_focuses_on_largest_saturated_component():
    pixels = np.full((100, 120, 3), 255, dtype=np.uint8)
    pixels[20:82, 34:76] = (30, 150, 45)
    cropped = LeafForegroundCrop()(Image.fromarray(pixels))
    assert cropped.width < 70
    assert cropped.height < 90
    assert cropped.width > 0 and cropped.height > 0


def test_leaf_foreground_crop_falls_back_for_uniform_background():
    image = Image.new("RGB", (80, 60), "white")
    assert LeafForegroundCrop()(image).size == image.size


def test_eval_transform_crop_ablation_preserves_model_input_shape():
    image = Image.new("RGB", (120, 100), "white")
    pixels = np.asarray(image).copy()
    pixels[15:90, 40:80] = (30, 150, 45)
    image = Image.fromarray(pixels)
    tensor = get_transforms(224, is_train=False, background_mode="leaf_crop")(image)
    assert tuple(tensor.shape) == (3, 224, 224)
