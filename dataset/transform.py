"""Image and mask transforms for semi-supervised segmentation.

Following ST++ patterns: all transforms take (img, mask) PIL pairs and return
transformed pairs. Masks use nearest interpolation to preserve label integrity.
"""

import math
import random

import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size):
    """Random crop of (img, mask) to (size, size)."""
    w, h = img.size
    padw = max(size - w, 0)
    padh = max(size - h, 0)

    if padw > 0 or padh > 0:
        img = transforms.functional.pad(img, (0, 0, padw, padh), fill=0)
        mask = transforms.functional.pad(mask, (0, 0, padw, padh), fill=255)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))
    return img, mask


def hflip(img, mask, p=0.5):
    """Random horizontal flip."""
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask):
    """Convert PIL images to tensors with ImageNet normalization.

    Returns:
        img: [3, H, W] float tensor, normalized
        mask: [H, W] long tensor
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    mask = torch.from_numpy(np.array(mask)).long()
    return img, mask


def resize(img, mask, base_size, ratio_range):
    """Random scale resize.

    Args:
        img, mask: PIL images
        base_size: reference size for scaling
        ratio_range: (min_ratio, max_ratio) tuple
    """
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]),
                               int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    """Random Gaussian blur on the image only."""
    if random.random() < p:
        sigma = random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_frac=(0.02, 0.4), ratio=(0.3, 3.3)):
    """Random rectangular cutout on both img and mask.

    Fills cutout region with zeros (img) and 255/ignore_index (mask).
    """
    if random.random() >= p:
        return img, mask

    img_array = np.array(img)
    mask_array = np.array(mask)
    h, w = img_array.shape[:2]

    area = h * w
    target_area = random.uniform(size_frac[0], size_frac[1]) * area
    aspect = random.uniform(ratio[0], ratio[1])

    cut_w = int(round(math.sqrt(target_area * aspect)))
    cut_h = int(round(math.sqrt(target_area / aspect)))
    cut_w = min(cut_w, w)
    cut_h = min(cut_h, h)

    x = random.randint(0, w - cut_w)
    y = random.randint(0, h - cut_h)

    img_array[y:y + cut_h, x:x + cut_w] = 0
    mask_array[y:y + cut_h, x:x + cut_w] = 255

    return Image.fromarray(img_array), Image.fromarray(mask_array)
