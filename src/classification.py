import torch
import torch.nn as nn
from torchvision.models import resnet18

from .utils import (
    get_device,
    image_to_tensor,
    mask_to_tensor,
    resize_with_pad_tensor,
    ensure_3ch_tensor,
    normalize_ultrasound_tensor
)


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.model = resnet18(weights=None)

        if in_channels != 3:
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.model(x)


def build_nodule_classifier(backbone: str = 'resnet18', num_classes: int = 2, in_channels: int = 3):
    return ResNetClassifier(num_classes=num_classes, in_channels=in_channels)


def crop_nodule_roi(image, mask=None, bbox=None, padding: int = 16):
    img = image_to_tensor(image)
    _, h, w = img.shape

    x1 = y1 = x2 = y2 = None

    if mask is not None:
        m = mask_to_tensor(mask)
        coords = torch.nonzero(m[0] > 0.5, as_tuple=False)

        if coords.numel() > 0:
            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item())
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item())
            x1, y1, x2, y2 = x_min, y_min, x_max, y_max

    if x1 is None and bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)

    if x1 is None:
        ch, cw = h // 2, w // 2
        size = min(h, w) // 2
        x1, y1 = max(0, cw - size // 2), max(0, ch - size // 2)
        x2, y2 = min(w, cw + size // 2), min(h, ch + size // 2)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return img[:, y1:y2, x1:x2]


def classify_nodule(model, roi, device: str = 'cuda'):
    device = get_device(device)
    model = model.to(device)
    model.eval()

    roi_t = image_to_tensor(roi)
    roi_t = resize_with_pad_tensor(roi_t, (224, 224), mode='bilinear')
    roi_t = ensure_3ch_tensor(roi_t)
    roi_t = normalize_ultrasound_tensor(roi_t)
    roi_t = roi_t.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(roi_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    return probs, pred


def extract_risk_features_from_mask(mask):
    """
    Extract simple prototype risk features from the predicted mask.
    """
    if mask is None:
        return {
            'taller_than_wide': False,
            'irregular_margin': False,
            'area': 0,
            'width': 0,
            'height': 0
        }

    m = mask_to_tensor(mask)[0]
    coords = torch.nonzero(m > 0.5, as_tuple=False)

    if coords.numel() == 0:
        return {
            'taller_than_wide': False,
            'irregular_margin': False,
            'area': 0,
            'width': 0,
            'height': 0
        }

    y_min = int(coords[:, 0].min().item())
    y_max = int(coords[:, 0].max().item())
    x_min = int(coords[:, 1].min().item())
    x_max = int(coords[:, 1].max().item())

    height = max(1, y_max - y_min + 1)
    width = max(1, x_max - x_min + 1)
    area = int((m > 0.5).sum().item())

    bbox_area = width * height
    fill_ratio = area / max(bbox_area, 1)

    taller_than_wide = height > width
    irregular_margin = fill_ratio < 0.65

    return {
        'taller_than_wide': taller_than_wide,
        'irregular_margin': irregular_margin,
        'area': area,
        'width': width,
        'height': height
    }


def compute_thyroid_risk_score(class_probs, mask=None, detections=None):
    """
    Prototype risk score in [0, 1].
    Based on:
    - malignant probability
    - taller-than-wide shape
    - irregular margin
    - optional detection confidence
    """
    if class_probs is None:
        return {
            'risk_score': 0.0,
            'risk_level': 'Unknown',
            'risk_features': {}
        }

    malignant_prob = float(class_probs[1]) if len(class_probs) > 1 else float(class_probs[0])
    features = extract_risk_features_from_mask(mask)

    score = 0.55 * malignant_prob

    if features['taller_than_wide']:
        score += 0.15

    if features['irregular_margin']:
        score += 0.15

    if detections is not None and len(detections) > 0:
        top_score = max([d['score'] for d in detections])
        score += 0.15 * float(top_score)

    score = max(0.0, min(score, 1.0))

    if score >= 0.70:
        risk_level = 'High'
    elif score >= 0.40:
        risk_level = 'Intermediate'
    else:
        risk_level = 'Low'

    return {
        'risk_score': round(score, 4),
        'risk_level': risk_level,
        'risk_features': features
    }