"""
THYROX - Main Pipeline
Complete workflow for thyroid nodule and lymph node detection
"""

import os
import sys
import argparse

sys.path.append('src')

from preprocessing import UltrasoundPreprocessor
from detection import ThyroidDetector
from classification import NoduleClassifier
from feature_extraction import FeatureExtractor
from ensemble import ThyroidEnsemble
from utils import Utils

def main(image_path, output_dir='results'):
    """Run complete THYROX pipeline on a single image"""
    
    print("=" * 60)
    print("THYROX - Thyroid Detection and Classification System")
    print("=" * 60)
    
    print("\n[1/6] Initializing components...")
    preprocessor = UltrasoundPreprocessor()
    detector = ThyroidDetector()
    classifier = NoduleClassifier()
    feature_extractor = FeatureExtractor()
    ensemble = ThyroidEnsemble()
    utils = Utils()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[2/6] Loading and preprocessing image...")
    try:
        processed_img = preprocessor.preprocess(image_path)
        print(f"  ✓ Image processed: {image_path}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return
    
    print("\n[3/6] Detecting nodules and lymph nodes...")
    detections = detector.detect(processed_img)
    print(f"  ✓ Found {len(detections)} regions of interest")
    
    if not detections:
        print("  ⚠ No nodules or lymph nodes detected")
        return
    
    print("\n[4/6] Analyzing detected regions...")
    results = []
    
    for i, det in enumerate(detections):
        print(f"\n  Region {i+1}: {det['label']}")
        print(f"    - Confidence: {det['confidence']:.3f}")
        print(f"    - BBox: {det['bbox']}")
        
        roi = preprocessor.extract_roi(processed_img, det['bbox'])
        
        classification = classifier.classify(roi)
        print(f"    - Classification: {classification['label']} ({classification['confidence']:.3f})")
        
        gray_roi = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
        margin_features = feature_extractor.extract_margin_features(binary)
        
        if margin_features:
            print(f"    - Margin irregularity: {margin_features['compactness']:.3f}")
        
        deep_features = classifier.get_deep_features(roi)
        
        combined = feature_extractor.combine_features(margin_features, deep_features)
        
        ensemble_result = ensemble.predict(combined)
        print(f"    - Ensemble: {ensemble_result['label']} ({ensemble_result['confidence']:.3f})")
        
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
    
    print("\n[5/6] Saving annotated result...")
    result_path = os.path.join(output_dir, 'result_' + os.path.basename(image_path))
    utils.save_result(
        processed_img,
        [r['detection'] for r in results],
        [r['classification'] for r in results],
        result_path
    )
    print(f"  ✓ Saved to: {result_path}")
    
    print("\n[6/6] Generating report...")
    report = utils.generate_report(
        image_path,
        [r['detection'] for r in results],
        [r['classification'] for r in results],
        [r['risk_score'] for r in results]
    )
    
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Report saved to: {report_path}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total regions detected: {len(results)}")
    print(f"High-risk regions: {sum(1 for r in results if r['risk_score'] > 0.7)}")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='THYROX - Thyroid Analysis')
    parser.add_argument('image', help='Path to ultrasound image')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    
    args = parser.parse_args()
    main(args.image, args.output)