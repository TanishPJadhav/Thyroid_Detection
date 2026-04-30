"""
THYROX - XGBoost Ensemble Module
Final classification using ensemble of detector outputs
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

class ThyroidEnsemble:
    def __init__(self, model_path='models/xgboost/ensemble.joblib'):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model_path = model_path
        self._train_placeholder()
    
    def _train_placeholder(self):
        """Create a simple placeholder model"""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.random.randint(0, 2, 200)
        self.model.fit(X, y)
    
    def predict(self, features):
        """Predict benign/malignant using ensemble"""
        if isinstance(features, list):
            features = np.array(features)
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Ensure correct size
        if features.shape[1] < 20:
            padded = np.zeros((1, 20))
            padded[0, :features.shape[1]] = features[0]
            features = padded
        elif features.shape[1] > 20:
            features = features[:, :20]
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        labels = ['benign', 'malignant']
        return {
            'class': int(prediction),
            'label': labels[int(prediction)],
            'confidence': float(probabilities[prediction]),
            'probabilities': probabilities.tolist()
        }
    
    def train(self, X_train, y_train):
        """Train ensemble model"""
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.feature_importances_