"""
THYROX - YOLOv5-based Detection Module
Detects thyroid nodules and suspicious lymph nodes
"""

import torch
import cv2
import numpy as np
from pathlib import Path

class ThyroidDetector:
    def __init__(self, model_path='models/yolov5/best.pt', conf_thresh=0.25, device='cpu'):
        self.conf_thresh = conf_thresh
        self.device = device
        
        # Load YOLOv5 model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=model_path, force_reload=True)
            self.model.to(device)
            self.model.conf = conf_thresh
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using placeholder detection")
            self.model = None
    
    def detect(self, image):
        """Detect nodules and lymph nodes in ultrasound image"""
        if self.model is None:
            return self._placeholder_detect(image)
        
        return self._yolo_detect(image)
    
    def _yolo_detect(self, image):
        """Real YOLO detection"""
        img_uint8 = (image * 255).astype(np.uint8)
        
        results = self.model(img_uint8)
        
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class': int(cls),
                'label': results.names[int(cls)]
            })
        
        return detections
    
    def _placeholder_detect(self, image):
        """Placeholder detection for demo"""
        h, w = image.shape[:2]
        
        detections = [
            {
                'bbox': [int(w*0.25), int(h*0.25), int(w*0.65), int(h*0.65)],
                'confidence': 0.87,
                'class': 0,
                'label': 'nodule'
            }
        ]
        
        if np.random.random() > 0.7:
            detections.append({
                'bbox': [int(w*0.1), int(h*0.6), int(w*0.3), int(h*0.8)],
                'confidence': 0.72,
                'class': 1,
                'label': 'lymph_node'
            })
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['label']
            
            color = (0, 255, 0) if label == 'nodule' else (255, 0, 0)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label}: {conf:.2f}"
            cv2.putText(img_copy, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_copy