"""
THYROX - Utility Functions
"""

import cv2
import numpy as np
from pathlib import Path

class Utils:
    @staticmethod
    def save_result(image, detections, classifications, output_path):
        """Save annotated result image"""
        if image.max() <= 1:
            img_copy = (image * 255).astype(np.uint8)
        else:
            img_copy = image.astype(np.uint8)
        
        if len(img_copy.shape) == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            
            color = (0, 255, 0) if label == 'nodule' else (255, 0, 0)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label}: {det['confidence']:.2f}"
            cv2.putText(img_copy, text, (x1, max(y1-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if classifications:
            cls = classifications[0]
            cls_text = f"Class: {cls['label']} ({cls['confidence']:.2f})"
            cv2.putText(img_copy, cls_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imwrite(output_path, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        return output_path
    
    @staticmethod
    def calculate_risk_score(detection_conf, classification_conf, margin_features=None):
        """Calculate overall risk score"""
        weights = {'detection': 0.3, 'classification': 0.5, 'margin': 0.2}
        
        score = (weights['detection'] * detection_conf + 
                weights['classification'] * classification_conf)
        
        if margin_features:
            irregularity = (margin_features.get('compactness', 0) + 
                          margin_features.get('tortuosity', 0)) / 2
            score += weights['margin'] * min(irregularity, 1.0)
        
        return min(score, 1.0)
    
    @staticmethod
    def generate_report(image_path, detections, classifications, risk_scores):
        """Generate text report"""
        report = []
        report.append("=" * 60)
        report.append("THYROX - Thyroid Analysis Report")
        report.append("=" * 60)
        report.append(f"Image: {image_path}")
        report.append("")
        
        for i, (det, cls, risk) in enumerate(zip(detections, classifications, risk_scores)):
            report.append(f"Region {i+1}:")
            report.append(f"  - Type: {det['label']}")
            report.append(f"  - Detection Confidence: {det['confidence']:.3f}")
            report.append(f"  - Classification: {cls['label']}")
            report.append(f"  - Classification Confidence: {cls['confidence']:.3f}")
            report.append(f"  - Risk Score: {risk:.3f}")
            report.append(f"  - Risk Level: {'HIGH' if risk > 0.7 else 'MEDIUM' if risk > 0.4 else 'LOW'}")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)