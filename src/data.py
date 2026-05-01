import os
import json
import copy
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .utils import image_to_tensor, mask_to_tensor, resize_with_pad_tensor, ensure_3ch_tensor, normalize_ultrasound_tensor


def load_thyroid_dataset(images_dir: str, annotations_path: str, split: str = 'train') -> list:
    """Load thyroid dataset or create dummy data for prototype."""
    samples = []
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            ann = json.load(f)
        entries = ann.get(split, [])
        for item in entries:
            image_path = os.path.join(images_dir, item['image_path'])
            mask_path = item.get('mask_path')
            mask_path = os.path.join(images_dir, mask_path) if mask_path else None
            img = np.array(Image.open(image_path).convert('L')) if os.path.exists(image_path) else np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            mask = np.array(Image.open(mask_path).convert('L')) if mask_path and os.path.exists(mask_path) else None
            samples.append({
                'image': img,
                'mask': mask,
                'bboxes': item.get('bboxes', None),
                'label': int(item.get('label', 0)),
                'meta': {'image_id': item.get('image_id', os.path.splitext(os.path.basename(image_path))[0]), 'patient_id': item.get('patient_id', 'unknown')}
            })
        return samples

    for i in range(8):
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cx, cy = np.random.randint(180, 460), np.random.randint(140, 340)
        rx, ry = np.random.randint(30, 70), np.random.randint(20, 55)
        yy, xx = np.ogrid[:480, :640]
        ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1
        mask[ellipse] = 1
        x1, y1 = max(0, cx - rx), max(0, cy - ry)
        x2, y2 = min(639, cx + rx), min(479, cy + ry)
        samples.append({
            'image': img,
            'mask': mask,
            'bboxes': [[x1, y1, x2, y2]],
            'label': np.random.randint(0, 2),
            'meta': {'image_id': f'{split}_{i}', 'patient_id': f'patient_{i//2}'}
        })
    return samples


def preprocess_image(image, target_size=(512, 512)):
    """Convert image to tensor, resize/pad, normalize."""
    image_tensor = image_to_tensor(image)
    image_tensor = resize_with_pad_tensor(image_tensor, target_size=target_size, mode='bilinear')
    image_tensor = ensure_3ch_tensor(image_tensor)
    image_tensor = normalize_ultrasound_tensor(image_tensor)
    return image_tensor


def augment_image(image, mask=None, bboxes=None):
    """Apply lightweight augmentation suitable for ultrasound images."""
    img = image_to_tensor(image)
    msk = mask_to_tensor(mask) if mask is not None else None
    boxes = copy.deepcopy(bboxes) if bboxes is not None else None
    _, h, w = img.shape
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.95, 1.05)
    brightness = random.uniform(0.9, 1.1)
    contrast = random.uniform(0.9, 1.1)
    img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
    if msk is not None:
        msk = TF.affine(msk, angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    if random.random() < 0.5:
        img = TF.hflip(img)
        if msk is not None:
            msk = TF.hflip(msk)
        if boxes is not None:
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]
    return img, msk, boxes