"""
THYROX - ResNet-based Classification Module
Classifies thyroid nodules as benign or malignant
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2

class NoduleClassifier:
    def __init__(self, model_path='models/resnet/best.pth', num_classes=2, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        
        # Load ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print("Using pretrained weights only")
        
        self.model.to(device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def classify(self, image):
        """Classify nodule as benign or malignant"""
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get prediction
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()
        
        labels = ['benign', 'malignant']
        return {
            'class': pred_class,
            'label': labels[pred_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
    
    def get_deep_features(self, image):
        """Extract deep features before final layer"""
        if len(image.shape) == 2:
            image = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Hook to extract features
        features = []
        def hook(module, input, output):
            features.append(output.flatten())
        
        handle = self.model.avgpool.register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(input_tensor)
        
        handle.remove()
        
        return features[0].cpu().numpy()