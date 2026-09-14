from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
import numpy as np

class BaselineAnomalyDetector:
    def __init__(self, random_state=42, contamination=0.15):
        # contamination reflects our estimated anomaly rate
        self.model = IsolationForest(
            n_estimators=150, 
            random_state=random_state, 
            contamination=contamination,
            n_jobs=-1
        )
    
    def fit(self, X):
        # We fill NaNs because scikit-learn's Isolation Forest cannot handle missing values natively.
        # Imputing with 0 ensures safe execution; missingness features already capture the NA information.
        self.model.fit(X.fillna(0))
        
    def predict(self, X):
        # Isolation Forest returns 1 for inliers, -1 for outliers
        # We map 1 -> 0 (normal) and -1 -> 1 (anomaly) to match ground truth
        preds = self.model.predict(X.fillna(0))
        return (preds == -1).astype(int)
        
    def save(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        
    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        
    def get_feature_importances(self, X, feature_names):
        """
        Calculates permutation importance since Isolation Forest 
        does not inherently expose feature_importances_.
        """
        from sklearn.inspection import permutation_importance
        # We use a dummy scoring function where we want the model to consistently predict its own outputs
        # This tells us which features are driving the model's decisions
        def score_func(estimator, X, y=None):
            preds = estimator.predict(X)
            # return mean anomaly score
            return -estimator.score_samples(X).mean()
            
        result = permutation_importance(self.model, X.fillna(0), np.zeros(len(X)), scoring=score_func, n_repeats=5, random_state=42, n_jobs=-1)
        importances = dict(zip(feature_names, result.importances_mean))
        # Sort descending
        return {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
