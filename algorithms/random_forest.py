import logging
from typing import DefaultDict, Union, Optional, List
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

logger = logging.getLogger("model")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S %p")

class RandomForestEstimator(ABC):
    '''
    Интерфейс для случайного леса для регрессии и классификации с OOB оценкой
    '''

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, 
                 max_features: str = "sqrt", 
                 criterion: Optional[str] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self._decision_trees = []
        self.oob_predictions = defaultdict(list)
        self.oob_score = None

    @abstractmethod
    def _init_decision_tree(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        pass

    @abstractmethod
    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X, self._y = X, y
        self.n, self.p = self._X.shape
        self._fit()
        
    def _fit(self):
        row_idx = list(range(self.n))
        for _ in range(self.n_estimators):
            sample_row_idx = np.random.choice(row_idx, size=self.n, replace=True)
            X_sampled, y_sampled = self._X[sample_row_idx, :], self._y[sample_row_idx]
            decision_tree = self._init_decision_tree()
            decision_tree.fit(X_sampled, y_sampled)
            self._decision_trees.append(decision_tree)
            oob_row_idx = list(set(row_idx) - set(sample_row_idx))
            if oob_row_idx:
                oob_prediction = decision_tree.predict(self._X[oob_row_idx])
                for idx, prediction in zip(oob_row_idx, oob_prediction):
                    self.oob_predictions[idx].append(prediction)
        self.oob_score = self._compute_oob_score(self._y, self.oob_predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for decision_tree in self._decision_trees:
            prediction = decision_tree.predict(X).reshape(-1,1)
            predictions.append(prediction)
        predictions = np.concatenate(predictions, axis=1)
        return self._compute_ensemble_prediction(predictions)

    @abstractmethod
    def _compute_ensemble_prediction(self, predictions: np.ndarray) -> np.ndarray:
        pass

class CustomRandomForestRegressor(RandomForestEstimator):
    def _init_decision_tree(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        return DecisionTreeRegressor(max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     max_features=self.max_features,
                                     criterion=self.criterion)

    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        y_true = np.array([y_true[idx] for idx in oob_predictions.keys()])
        y_oob = np.array([np.mean(predictions) for predictions in oob_predictions.values()])
        return r2_score(y_true, y_oob)

    def _compute_ensemble_prediction(self, predictions: np.ndarray) -> np.ndarray:
        return np.mean(predictions, axis=1)
    
class CustomRandomForestClassifier(RandomForestEstimator):
    def _init_decision_tree(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        return DecisionTreeClassifier(max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      max_features=self.max_features,
                                      criterion=self.criterion)

    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        y_true = np.array([y_true[idx] for idx in oob_predictions.keys()])
        y_oob = np.array([np.bincount(predictions).argmax() for predictions in oob_predictions.values()])
        return accuracy_score(y_true, y_oob)

    def _compute_ensemble_prediction(self, predictions: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions.astype(int))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for decision_tree in self._decision_trees:
            prediction = decision_tree.predict(X).reshape(-1,1)
            predictions.append(prediction)
        predictions = np.concatenate(predictions, axis=1)
        return np.mean(predictions, axis=1)