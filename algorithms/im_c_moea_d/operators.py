"""IM-C-MOEA-D 算子与聚类模块 (对应 Operator.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array


def kmeans_clusters(objs: Array, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-Means 算法归类，返回样本所属类别标签 (N,)。"""
    n = len(objs)
    if n <= k:
        return np.arange(n)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = objs[init_idx].copy()

    for _ in range(15):
        dist = np.linalg.norm(objs[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(k):
            members = objs[labels == i]
            new_centers[i] = np.mean(members, axis=0) if len(members) > 0 else centers[i]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels
