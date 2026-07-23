"""DRLOS-EMCMO Dropout 回归与 Q 网络算法模块 (对应 Dropout 文件夹)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array


class SimpleQNet:
    """NumPy 实现的简单 Q-学习/函数拟合网络 (对应 Dropout 模型)。"""

    def __init__(self, in_features: int = 4, hidden: int = 10, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        self.w1 = rng.normal(0, 0.1, size=(in_features, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)

    def predict(self, x: Array) -> Array:
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.dot(h, self.w2) + self.b2
        return out.ravel()

    def fit(self, x: Array, y: Array, epochs: int = 20, lr: float = 0.01):
        for _ in range(epochs):
            # Forward
            h_in = np.dot(x, self.w1) + self.b1
            h = np.tanh(h_in)
            out = (np.dot(h, self.w2) + self.b2).ravel()

            # Backward
            err = out - y
            d_out = (2.0 * err / len(x))[:, None]
            d_w2 = np.dot(h.T, d_out)
            d_b2 = np.sum(d_out, axis=0)

            d_h = np.dot(d_out, self.w2.T) * (1.0 - h**2)
            d_w1 = np.dot(x.T, d_h)
            d_b1 = np.sum(d_h, axis=0)

            self.w2 -= lr * d_w2
            self.b2 -= lr * d_b2
            self.w1 -= lr * d_w1
            self.b1 -= lr * d_b1
