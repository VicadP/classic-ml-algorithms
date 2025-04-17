import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union


logger = logging.getLogger("model")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S %p")

class CustomSVMClassifier():
    '''
    SVM с hinge loss и обучением через стохастический градиентный спуск
    '''
    def __init__(self, 
                 n_iter: int = 1000, 
                 learning_rate: float = 0.001, 
                 alpha: float = 1.0, 
                 tolerance: float = 1e-3, 
                 verbose = False):
        self.n_iter = n_iter
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.alpha = 1 / alpha
        self.tolerance = tolerance
        self.verbose = verbose
        self.objective_path = []
        self.weights_path = []

    @staticmethod
    def add_constant(X: np.ndarray) -> np.ndarray:
        intercept_column = np.ones(X.shape[0]).reshape(-1,1)
        return np.hstack((intercept_column, X))
    
    def _transform_taget(self, y: np.ndarray) -> np.ndarray:
        return np.where(y == 0, -1, 1)
    
    def _inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        return np.where(y == -1, 0, 1)
    
    def _compute_margin(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return y * np.dot(x, weights)

    def _compute_loss(self, weights: np.ndarray) -> float:
        loss_orig = self.alpha * np.mean(np.maximum(0, 1 - self._compute_margin(self._X, self._y, weights)))
        reg_term = 0.5 * np.dot(weights, weights)
        return loss_orig + reg_term

    def _compute_gradient(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        dist_to_hyperplan = 1 - self._compute_margin(x, y, weights)
        if dist_to_hyperplan >= 1:
            return weights
        else:
            return weights - self.alpha * y * x

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X, self._y = self.add_constant(X), self._transform_taget(y)
        self.n, self.p = self._X.shape
        self.weights_ = self._fit()

    def _fit(self) -> np.ndarray:
        weights = np.random.normal(loc=0, scale=0.1, size=self.p)
        self.objective_path.append(self._compute_loss(weights))   # считаем лосс на всех данных
        self.weights_path.append(weights)
        for epoch in range(self.n_iter):
            idx = np.random.permutation(self.n)
            X_shuffled = self._X[idx]
            y_shuffled = self._y[idx]
            for i in range(self.n):
                step = self.learning_rate * self._compute_gradient(X_shuffled[i], y_shuffled[i], weights)
                weights = weights - step
            self.learning_rate = self.initial_learning_rate / (epoch + 1)
            self.weights_path.append(weights)
            self.objective_path.append(self._compute_loss(self.weights_path[-1]))
            if np.linalg.norm(self.weights_path[-1] - self.weights_path[-2]) < self.tolerance:
                logger.info(f'Алгоритм сошелся. Кол-во итераций: {epoch}')
                break
            if self.verbose == True and epoch % 10 == 0:
                logger.info(f'Итерация: {epoch}; Функционал ошибки: {self._compute_loss(self.weights_path[-1])}')
        return self.weights_path[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.add_constant(X)
        pred = np.sign(np.dot(X, self.weights_))
        return self._inverse_transform_target(pred)