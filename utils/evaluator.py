from typing import List
import logging
import numpy as np
import pandas as pd
from utils.metrics import Metric
from sklearn.model_selection import train_test_split

logger = logging.getLogger('evaluator')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p')

class ModelEvaluator():
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 test_size: float = 0.3, 
                 shuffle: bool = True, 
                 random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
                                                                                test_size=test_size, 
                                                                                shuffle=shuffle, 
                                                                                random_state=random_state)
        self.results = []

    def evaluate_model(self, model, model_name: str, metrics: List[Metric], predict_proba: bool = False):
        model.fit(self.X_train, self.y_train)
        if predict_proba:
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            if y_pred_proba.ndim == 2: # для обработки предсказаний sklearn
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred = model.predict(self.X_test)
        metrics_dict = {}
        for metric in metrics:
            if metric.get_name() == "ROC AUC":
                metric_value = metric.compute_value(self.y_test, y_pred_proba)
            else:
                metric_value = metric.compute_value(self.y_test, y_pred)
            metrics_dict[metric.get_name()] = metric_value
        self.results.append({"model": model_name, **metrics_dict})
    
    def get_result(self) -> pd.DataFrame:
        result = pd.DataFrame(self.results)
        return result.set_index("model")