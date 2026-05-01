import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .utils import get_device, image_to_tensor, ensure_3ch_tensor, resize_with_pad_tensor


def build_nodule_detector(backbone: str = 'fasterrcnn_resnet50', num_classes: int = 2):
    if backbone != 'fasterrcnn_resnet50':
        raise ValueError('Currently supported: fasterrcnn_resnet50')
    detector = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return detector


def detect_nodules(detector, image, score_threshold: float = 0.3, device: str = 'cuda') -> list:
    device = get_device(device)
    detector = detector.to(device)
    detector.eval()
    img = image_to_tensor(image)
    img = resize_with_pad_tensor(img, target_size=(512, 512), mode='bilinear')
    img = ensure_3ch_tensor(img).to(device)
    with torch.no_grad():
        outputs = detector([img])[0]
    detections = []
    for b, s, l in zip(outputs['boxes'].detach().cpu().numpy(), outputs['scores'].detach().cpu().numpy(), outputs['labels'].detach().cpu().numpy()):
        if float(s) >= score_threshold:
            detections.append({'bbox': b.tolist(), 'score': float(s), 'label': int(l)})
    return detections