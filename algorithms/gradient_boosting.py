import logging
from typing import (Union, Optional, List)
import numpy as np
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

class GradientBoostingEstimator(ABC):
    '''
    Интерфейс для градиентного бустинга для классификации и регрессии
    '''
    def __init__(self, 
                 learning_rate: float = 0.1, 
                 n_estimators: int = 100, 
                 min_samples_split: int = 2, 
                 max_depth: int = 3, 
                 criterion: Optional[str] = None):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self._decision_trees = []

    @abstractmethod
    def _init_decision_tree(self) -> DecisionTreeRegressor:
        pass

    @abstractmethod
    def _compute_initial_pred(self, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def _update_leaf_nodes(self, 
                           tree: DecisionTreeRegressor, 
                           X: np.ndarray, y: np.ndarray, 
                           residuals: np.ndarray):
        pass

    @abstractmethod
    def _transform_ensemble_pred(self, ensemble_pred: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.initial_pred = self._compute_initial_pred(y)
        ensemble_pred = np.full(X.shape[0], self.initial_pred)
        for _ in range(self.n_estimators):
            residuals = y - self._transform_ensemble_pred(ensemble_pred)
            decision_tree = self._init_decision_tree()
            decision_tree.fit(X, residuals)
            self._update_leaf_nodes(decision_tree.tree_, X, y, residuals)
            self._decision_trees.append(decision_tree)
            ensemble_pred += decision_tree.predict(X)

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class CustomGradientBoostingRegressor(GradientBoostingEstimator):
    def _init_decision_tree(self)  -> DecisionTreeRegressor:
        return DecisionTreeRegressor(max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     criterion=self.criterion)

    def _compute_initial_pred(self, y: np.ndarray) -> float:
        return np.mean(y)

    def _update_leaf_nodes(self, 
                           tree: DecisionTreeRegressor, 
                           X: np.ndarray, y: np.ndarray, 
                           residuals: np.ndarray):
        X = X.astype(np.float32)
        leafs = tree.apply(X)
        for leaf in np.unique(leafs):
            tree.value[leaf, 0, 0] *= self.learning_rate          

    def _transform_ensemble_pred(self, ensemble_pred: np.ndarray) -> np.ndarray:
        return ensemble_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_pred)
        for i in range(self.n_estimators):
            predictions += self._decision_trees[i].predict(X)
        return predictions

class CustomGradientBoostingClassifier(GradientBoostingEstimator):
    def _init_decision_tree(self)  -> DecisionTreeRegressor:
        return DecisionTreeRegressor(max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     criterion=self.criterion)

    def _compute_initial_pred(self, y: np.ndarray) -> float:
        return 0. # log(odds) = 0 ==> proba = 0.5

    def _update_leaf_nodes(self, 
                           tree: DecisionTreeRegressor, 
                           X: np.ndarray, y: np.ndarray, 
                           residuals: np.ndarray):
        X = X.astype(np.float32)
        leafs = tree.apply(X)
        for leaf in np.unique(leafs):
            idx = np.nonzero(leafs == leaf)[0]
            p = y[idx] - residuals[idx]
            assert not np.any((p < 0) | (p > 1)), "вероятность вне допустимого диапазона [0,1] при расчете значения листа"
            numerator = np.mean(residuals[idx])
            denominator = np.mean(p * (1 - p))
            gamma = numerator / denominator
            tree.value[leaf, 0, 0] = self.learning_rate * gamma

    def _transform_ensemble_pred(self, ensemble_pred: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-ensemble_pred)) # приводим логарифм шансов к вероятности через сигмоиду

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_pred) # предсказываем логарифм шансов
        for i in range(self.n_estimators):
            predictions += self._decision_trees[i].predict(X)
        return self._transform_ensemble_pred(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_pred)
        for i in range(self.n_estimators):
            predictions += self._decision_trees[i].predict(X)
        return (predictions > 0.).astype(int)