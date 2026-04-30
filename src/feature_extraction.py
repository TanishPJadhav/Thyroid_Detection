"""
THYROX - Feature Extraction Module
Combines margin geometric features with deep CNN features
"""

import numpy as np
import cv2
from skimage import measure, morphology
from scipy.spatial import ConvexHull

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_margin_features(self, roi_mask):
        """Extract 9 geometric margin features"""
        labeled = measure.label(roi_mask)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return None
        
        region = regions[0]
        features = {}
        
        area = region.area
        perimeter = region.perimeter
        convex_area = region.convex_area
        bbox = region.bbox
        
        # 1. Circularity
        features['circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # 2. Compactness
        features['compactness'] = 1 - (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 3. Convexity
        features['convexity'] = region.convex_image.sum() / perimeter if perimeter > 0 else 0
        
        # 4. Solidity
        features['solidity'] = area / convex_area if convex_area > 0 else 0
        
        # 5. Dispersion
        major_axis = region.major_axis_length
        features['dispersion'] = major_axis / area if area > 0 else 0
        
        # 6. Ratio Aspect
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
    
    def combine_features(self, margin_features, deep_features):
        """Combine margin and deep features into single vector"""
        if margin_features is None:
            return deep_features
        
        margin_vector = np.array([
            margin_features['circularity'],
            margin_features['compactness'],
            margin_features['convexity'],
            margin_features['solidity'],
            margin_features['dispersion'],
            margin_features['ratio_aspect'],
            margin_features['tortuosity'],
            margin_features['ratio_hw'],
            margin_features['rectangularity']
        ])
        
        return np.concatenate([margin_vector, deep_features])
    
    def get_feature_names(self):
        """Return feature names for reference"""
        return [
            'circularity', 'compactness', 'convexity', 'solidity',
            'dispersion', 'ratio_aspect', 'tortuosity', 'ratio_hw', 'rectangularity'
        ]