"""DVCEA 适应度计算模块 (对应 CalFitness.m 与 CalFitness_E.m)。"""
from __future__ import annotations

import math
import numpy as np
from core.schema import Array


def cal_fitness_e(objs: Array, cv: Array, var_epsilon: float) -> Array:
    """计算考虑 epsilon 的 SPEA2 适应度 (对应 CalFitness_E.m)。"""
    n = len(objs)
    eps_cv = np.maximum(0.0, cv - var_epsilon)

    dominate = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if eps_cv[i] < eps_cv[j]:
                dominate[i, j] = True
            elif eps_cv[i] > eps_cv[j]:
                dominate[j, i] = True
            else:
                less = np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j])
                more = np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i])
                if less:
                    dominate[i, j] = True
                elif more:
                    dominate[j, i] = True

    s = np.sum(dominate, axis=1)
    r = np.zeros(n, dtype=float)
    for i in range(n):
        r[i] = np.sum(s[dominate[:, i]])

    dist = np.linalg.norm(objs[:, None, :] - objs[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    dist_sort = np.sort(dist, axis=1)
    k_idx = max(0, int(math.floor(math.sqrt(n))) - 1)
    d = 1.0 / (dist_sort[:, k_idx] + 2.0)

    return r + d
