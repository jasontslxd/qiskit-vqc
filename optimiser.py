from abc import ABC, abstractmethod
import numpy as np

class Optimiser(ABC):
    def __init__(self, lr=0.01):
        self.lr = lr

    @abstractmethod
    def step(self) -> np.ndarray:
        pass

class Adam(Optimiser):
    def __init__(self, lr=0.01, beta1:float=0.9, beta2:float=0.999, eps:float=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, parameters:np.ndarray, gradient_vector:np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros(parameters.shape[0])
            self.v = np.zeros(parameters.shape[0])
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient_vector
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient_vector ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return parameters - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)