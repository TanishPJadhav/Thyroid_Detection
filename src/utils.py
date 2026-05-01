import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = 'cuda') -> torch.device:
    if device == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device)


def image_to_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        img = image.float()
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.max() > 1.0:
            img = img / 255.0
        return img
    if isinstance(image, Image.Image):
        arr = np.array(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f'Unsupported image type: {type(image)}')
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    img = torch.from_numpy(arr).permute(2, 0, 1).float()
    if img.max() > 1.0:
        img = img / 255.0
    return img


def mask_to_tensor(mask):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        m = mask.float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        return (m > 0).float()
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (torch.from_numpy(mask).float() > 0).float().unsqueeze(0)
    raise TypeError(f'Unsupported mask type: {type(mask)}')


def ensure_3ch_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    elif image_tensor.shape[0] > 3:
        image_tensor = image_tensor[:3]
    return image_tensor


def resize_with_pad_tensor(tensor: torch.Tensor, target_size=(512, 512), mode: str = 'bilinear') -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    _, h, w = tensor.shape
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = tensor.unsqueeze(0)
    if mode == 'nearest':
        x = F.interpolate(x, size=(new_h, new_w), mode=mode)
    else:
        x = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=False)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return x.squeeze(0)


def normalize_ultrasound_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = ensure_3ch_tensor(image_tensor)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor - mean) / std