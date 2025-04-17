import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

logger = logging.getLogger("model")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S %p")

class LinearEstimator(ABC):
    '''
    Интерфейс для линейной и логистической регрессии с оптимизацией через градиентный спуск и возможностью задать динамическую скорость обучения
    '''
    def __init__(self, 
                 n_iter: int = 1000, 
                 learning_rate: Union[float, Callable] = 0.1, 
                 tolerance: float = 1e-4,
                 penalty: Optional[str] = None, 
                 alpha: float = 1.0, 
                 verbose: bool = False):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        assert penalty in [None, "L1", "L2"], "Некорректный ввод. Допустимые значения: None, L1, L2"
        self.penalty = penalty
        self.alpha = alpha
        self.verbose = verbose
        self.objective_path = []    
        self.weights_path = []  

    @staticmethod
    def add_constant(X: np.ndarray) -> np.ndarray:
        intercept_column = np.ones(X.shape[0]).reshape(-1,1)
        return np.hstack((intercept_column, X))

    @abstractmethod
    def _compute_loss(self, weights: np.ndarray) -> float:
        pass

    @abstractmethod
    def _compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X, self._y = self.add_constant(X), y
        self.n, self.p = self._X.shape
        self.weights_ = self._fit()
        
    def _fit(self) -> np.ndarray:
        weights = np.random.normal(loc=0, scale=1, size=self.p)
        self.objective_path.append(self._compute_loss(weights))
        self.weights_path.append(weights)
        for epoch in range(self.n_iter):
            if callable(self.learning_rate):
                step = self.learning_rate(epoch) * self._compute_gradient(weights)
            else:
                step = self.learning_rate * self._compute_gradient(weights)
            weights = weights - step
            if np.linalg.norm(weights - self.weights_path[-1]) < self.tolerance:
                logger.info(f'Алгоритм сошелся. Кол-во итераций: {epoch}')
                break
            self.weights_path.append(weights)
            if self.penalty == "L1":
                self.weights_path[-1][self.weights_path[-1] < self.tolerance] = 0
            self.objective_path.append(self._compute_loss(self.weights_path[-1]))
            if self.verbose == True and epoch % 10 == 0:
                logger.info(f'Итерация: {epoch}; Функционал ошибки: {self._compute_loss(self.weights_path[-1])}')
        return self.weights_path[-1]

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class CustomLinearRegression(LinearEstimator):
    def mse(self, weights: np.ndarray) -> float:
        y_pred = np.dot(self._X, weights)
        return np.mean(np.square(self._y - y_pred))
    
    def mse_grad(self, weights: np.ndarray) -> np.ndarray:
        y_pred = np.dot(self._X, weights)
        return np.dot(self._X.T, (y_pred - self._y)) / self.n

    def _compute_loss(self, weights: np.ndarray) -> float:
        loss_orig = self.mse(weights)        
        if self.penalty is None:
            return loss_orig
        elif self.penalty == "L1":
            reg_term: float = self.alpha * np.sum(np.abs(weights[1:]))
            return loss_orig + reg_term
        elif self.penalty == "L2":
            reg_term: float = (self.alpha / self.n) * np.sum(np.square(weights[1:]))
            return loss_orig + reg_term

    def _compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        loss_orig_grad = self.mse_grad(weights)
        if self.penalty is None:
            return loss_orig_grad
        elif self.penalty == "L1":
            reg_term_grad = np.hstack([0, self.alpha * np.sign(weights[1:])])
            return loss_orig_grad + reg_term_grad
        elif self.penalty == "L2":
            reg_term_grad = np.hstack([0, (self.alpha / self.n) * weights[1:]])
            return loss_orig_grad + reg_term_grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.add_constant(X)
        return np.dot(X, self.weights_)
    
class CustomLogisticRegression(LinearEstimator):
    @staticmethod
    def sigmoid(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.dot(X, weights)))
    
    def bce(self, weights: np.ndarray) -> float:
        y_pred = self.sigmoid(self._X, weights) + 1e-9
        return -np.mean(self._y * np.log(y_pred) + (1 - self._y) * np.log(1 - y_pred))
    
    def bce_grad(self, weights: np.ndarray) -> np.ndarray:
        y_pred = self.sigmoid(self._X, weights)
        return np.dot(self._X.T, (y_pred - self._y)) / self.n

    def _compute_loss(self, weights: np.ndarray) -> float:
        loss_orig = self.bce(weights)
        if self.penalty is None:
            return loss_orig
        if self.penalty == "L1":
            pass    # не реализовывал, т.к. идентично линейной регрессии      
        if self.penalty == "L2":
            pass

    def _compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        loss_orig_grad = self.bce_grad(weights)
        if self.penalty is None:
            return loss_orig_grad
        elif self.penalty == "L1":
            pass
        elif self.penalty == "L2":
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.add_constant(X)
        proba = self.sigmoid(X, self.weights_)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.add_constant(X)
        return self.sigmoid(X, self.weights_)