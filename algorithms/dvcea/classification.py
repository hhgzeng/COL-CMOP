"""DVCEA 决策变量分类模块 (对应 Variable_classification.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import CMOP, Array, Population


def kmeans_clusters(x: Array, k: int, rng: np.random.Generator) -> Array:
    """简易 K-Means 聚类，返回中心点 (K, D)。"""
    n, d = x.shape
    if n <= k:
        return x.copy()
    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()
    for _ in range(10):
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(k):
            members = x[labels == i]
            new_centers[i] = np.mean(members, axis=0) if len(members) > 0 else centers[i]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def variable_classification(
    problem: CMOP, pop: Population, centers: Array, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """决策变量分类 (FEA: 可行性主导, INFEA: 收敛性主导)。"""
    d = pop.x.shape[1]
    if len(centers) < 2:
        fea = np.arange(d // 2, dtype=int)
        infea = np.arange(d // 2, d, dtype=int)
        return fea, infea

    # 扰动法评估变量对约束与目标的影响度
    scores = np.zeros(d, dtype=float)
    base_x = centers[0]
    base_eval = problem.evaluate(base_x[None, :])
    base_cv = base_eval.cv[0]

    for i in range(d):
        perturbed_x = base_x.copy()
        span = problem.upper[i] - problem.lower[i]
        perturbed_x[i] = np.clip(perturbed_x[i] + 0.1 * span, problem.lower[i], problem.upper[i])
        pert_eval = problem.evaluate(perturbed_x[None, :])
        scores[i] = abs(pert_eval.cv[0] - base_cv)

    median_score = np.median(scores)
    fea = np.flatnonzero(scores >= median_score)
    infea = np.flatnonzero(scores < median_score)

    if len(fea) == 0:
        fea = np.arange(d // 2, dtype=int)
        infea = np.arange(d // 2, d, dtype=int)
    elif len(infea) == 0:
        infea = np.arange(d // 2, dtype=int)
        fea = np.arange(d // 2, d, dtype=int)

    return fea, infea
