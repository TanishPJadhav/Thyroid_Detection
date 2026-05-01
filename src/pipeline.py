from .data import preprocess_image
from .segmentation import predict_nodule_mask
from .detection import detect_nodules
from .classification import (
    crop_nodule_roi,
    classify_nodule,
    compute_thyroid_risk_score
)


def run_full_thyroid_pipeline(image, segmentation_model=None, detector=None, classifier=None, device: str = 'cuda') -> dict:
    _ = preprocess_image(image, target_size=(512, 512))

    pred_mask = None
    detections = None
    roi = None
    class_probs = None
    predicted_class = None
    risk_output = None

    if segmentation_model is not None:
        pred_mask = predict_nodule_mask(segmentation_model, image, device=device)
        roi = crop_nodule_roi(image, mask=pred_mask, bbox=None, padding=16)

    elif detector is not None:
        detections = detect_nodules(detector, image, score_threshold=0.3, device=device)
        if len(detections) > 0:
            best_det = sorted(detections, key=lambda x: x['score'], reverse=True)[0]
            roi = crop_nodule_roi(image, mask=None, bbox=best_det['bbox'], padding=16)

    if classifier is not None and roi is not None:
        class_probs, predicted_class = classify_nodule(classifier, roi, device=device)
        risk_output = compute_thyroid_risk_score(
            class_probs=class_probs,
            mask=pred_mask,
            detections=detections
        )

    return {
        'mask': pred_mask,
        'detections': detections,
        'roi': roi,
        'class_probs': class_probs,
        'predicted_class': predicted_class,
        'risk_score': None if risk_output is None else risk_output['risk_score'],
        'risk_level': None if risk_output is None else risk_output['risk_level'],
        'risk_features': None if risk_output is None else risk_output['risk_features']
    }