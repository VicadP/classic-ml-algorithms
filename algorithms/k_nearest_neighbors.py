import logging
from abc import ABC, abstractmethod
import numpy as np
import faiss

logger = logging.getLogger("model")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S %p")

class KNNEstimator(ABC):
    '''
    Интерфейс для метода ближайших соседей для классификации и регрессии.
    Для ускорения обучения и предсказания используется библиотека faiss
    '''

    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = k
        assert weights in ["uniform", "distance"], "Некорректный ввод, допустимые значения 'uniform', 'distance'"
        self.weights = weights
        self.index = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.index = faiss.IndexFlatL2(X.shape[1]) 
        self.index.add(X.astype(np.float32))
        self._y = y

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        if self.weights == 'uniform':
            return np.ones_like(distances)
        if self.weights == 'distance':
            return 1 / (distances + 1e-9)

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class CustomKNNRegressor(KNNEstimator):
    def predict(self, X: np.ndarray) -> np.ndarray:
        distances, idx = self.index.search(X.astype(np.float32), k=self.k)
        weights = self._compute_weights(distances)
        return np.sum(self._y[idx] * weights, axis=1) / np.sum(weights, axis=1)
    
class CustomKNNClassifiers(KNNEstimator):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances, idx = self.index.search(X.astype(np.float32), k=self.k)
        weights = self._compute_weights(distances)
        return np.sum(self._y[idx] * weights, axis=1) / np.sum(weights, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)