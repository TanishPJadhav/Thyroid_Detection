"""
THYROX - Ultrasound Image Preprocessing Module
Handles: resizing, normalization, denoising, ROI extraction
"""

import cv2
import numpy as np
from skimage import measure, filters, morphology
from skimage.segmentation import active_contour
import torch
from torchvision import transforms

class UltrasoundPreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def load_image(self, image_path):
        """Load ultrasound image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def denoise(self, image):
        """Apply speckle noise reduction using bilateral filter"""
        adaptive_median = cv2.medianBlur(image, 7)
        denoised = cv2.bilateralFilter(adaptive_median, 9, 75, 75)
        return denoised
    
    def enhance_contrast(self, image):
        """CLAHE for contrast enhancement"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def resize(self, image):
        """Resize to target size"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_intensity(self, image):
        """Normalize pixel values to [0, 1]"""
        return image.astype(np.float32) / 255.0
    
    def extract_roi(self, image, bbox):
        """Extract region of interest using bounding box"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]
    
    def preprocess(self, image_path):
        """Full preprocessing pipeline"""
        image = self.load_image(image_path)
        image = self.denoise(image)
        image = self.enhance_contrast(image)
        image = self.resize(image)
        image = self.normalize_intensity(image)
        return image
    
    def to_tensor(self, image):
        """Convert to PyTorch tensor"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        return transform(image)


class ActiveContourSegmenter:
    def __init__(self):
        pass
    
    def segment(self, image, init_mask):
        """Active contour without edges (Chan-Vese)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        s = np.linspace(0, 2*np.pi, 400)
        r = init_mask[0] + init_mask[2]*np.sin(s)
        c = init_mask[1] + init_mask[3]*np.cos(s)
        init = np.array([r, c]).T
        
        snake = active_contour(gray, init, alpha=0.015, beta=10, gamma=0.001)
        
        return snake


class MarginFeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, binary_mask):
        """Extract 9 geometric margin features"""
        labeled = measure.label(binary_mask)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return None
        
        region = regions[0]
        
        features = {}
        
        perimeter = region.perimeter
        area = region.area
        
        # 1. Circularity
        features['circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # 2. Compactness
        features['compactness'] = 1 - (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 3. Convexity
        features['convexity'] = region.convex_area / perimeter if perimeter > 0 else 0
        
        # 4. Solidity
        features['solidity'] = area / region.convex_area if region.convex_area > 0 else 0
        
        # 5. Dispersion
        major_axis = region.major_axis_length
        features['dispersion'] = major_axis / area if area > 0 else 0
        
        # 6. Ratio Aspect
        bbox = region.bbox
        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]
        features['ratio_aspect'] = width / height if height > 0 else 0
        
        # 7. Tortuosity
        features['tortuosity'] = 2 * major_axis / perimeter if perimeter > 0 else 0
        
        # 8. Ratio Height-Width
        bbox_area = height * width
        features['ratio_hw'] = area / bbox_area if bbox_area > 0 else 0
        
        # 9. Rectangularity
        features['rectangularity'] = area / bbox_area if bbox_area > 0 else 0
        
        return features