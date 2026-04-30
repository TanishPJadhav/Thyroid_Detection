"""
THYROX - Demo Script
Run a demonstration of the complete pipeline with sample data
"""

import os
import sys
import numpy as np
import cv2

sys.path.append('src')

from preprocessing import UltrasoundPreprocessor
from detection import ThyroidDetector
from classification import NoduleClassifier
from feature_extraction import FeatureExtractor
from ensemble import ThyroidEnsemble
from utils import Utils

def create_sample_ultrasound_image():
    """Create a synthetic ultrasound-like image for demo"""
    img = np.random.normal(128, 20, (512, 512)).astype(np.uint8)
    
    center = (256, 256)
    axes = (80, 60)
    angle = 30
    cv2.ellipse(img, center, axes, angle, 0, 360, 80, -1)
    
    noise = np.random.normal(0, 10, (512, 512)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb

def run_demo():
    """Run complete demo pipeline"""
    
    print("=" * 70)
    print("THYROX - Thyroid Detection and Classification System (DEMO)")
    print("=" * 70)
    
    print("\n[1/6] Creating sample ultrasound image...")
    sample_img = create_sample_ultrasound_image()
    
    os.makedirs('demo', exist_ok=True)
    sample_path = 'demo/sample_ultrasound.jpg'
    cv2.imwrite(sample_path, cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Sample image created: {sample_path}")
    
    print("\n[2/6] Initializing THYROX components...")
    preprocessor = UltrasoundPreprocessor()
    detector = ThyroidDetector()
    classifier = NoduleClassifier()
    feature_extractor = FeatureExtractor()
    ensemble = ThyroidEnsemble()
    utils = Utils()
    print("  ✓ All components initialized")
    
    print("\n[3/6] Preprocessing image...")
    processed = preprocessor.preprocess(sample_path)
    print(f"  ✓ Image preprocessed: shape={processed.shape}")
    
    print("\n[4/6] Detecting regions of interest...")
    detections = detector.detect(processed)
    print(f"  ✓ Detected {len(detections)} regions")
    
    for i, det in enumerate(detections):
        print(f"    Region {i+1}: {det['label']} (conf: {det['confidence']:.3f})")
    
    print("\n[5/6] Analyzing detected regions...")
    results = []
    
    for i, det in enumerate(detections):
        print(f"\n  Region {i+1} Analysis:")
        
        roi = preprocessor.extract_roi(processed, det['bbox'])
        
        classification = classifier.classify(roi)
        print(f"    - CNN: {classification['label']} ({classification['confidence']:.3f})")
        
        gray_roi = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
        margin_features = feature_extractor.extract_margin_features(binary)
        
        if margin_features:
            print(f"    - Margin Features:")
            for feat_name, feat_val in margin_features.items():
                print(f"      {feat_name}: {feat_val:.4f}")
        
        deep_features = classifier.get_deep_features(roi)
        print(f"    - Deep features: {len(deep_features)} dimensions")
        
        combined = feature_extractor.combine_features(margin_features, deep_features)
        print(f"    - Combined features: {len(combined)} dimensions")
        
        ensemble_result = ensemble.predict(combined)
        print(f"    - Ensemble prediction: {ensemble_result['label']}")
        print(f"    - Ensemble confidence: {ensemble_result['confidence']:.3f}")
        
        risk = utils.calculate_risk_score(
            det['confidence'],
            classification['confidence'],
            margin_features
        )
        print(f"    - Risk Score: {risk:.3f}")
        
        results.append({
            'detection': det,
            'classification': classification,
            'ensemble': ensemble_result,
            'risk_score': risk,
            'margin_features': margin_features
        })
    
    print("\n[6/6] Saving results...")
    
    result_path = 'demo/result_annotated.jpg'
    utils.save_result(
        processed,
        [r['detection'] for r in results],
        [r['classification'] for r in results],
        result_path
    )
    print(f"  ✓ Annotated image: {result_path}")
    
    report = utils.generate_report(
        sample_path,
        [r['detection'] for r in results],
        [r['classification'] for r in results],
        [r['risk_score'] for r in results]
    )
    
    report_path = 'demo/report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Report: {report_path}")
    
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    
    print("\n✓ Demo completed successfully!")
    print("Check demo/ directory for outputs")

if __name__ == '__main__':
    run_demo()