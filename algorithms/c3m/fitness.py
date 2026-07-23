"""C3M 适应度计算模块 (对应 CalFitness.m)。"""
from __future__ import annotations

import math
import numpy as np
from core.schema import Array


def cal_fitness(objs: Array, g: Array | None, processcon: int = 0, totalcon: int = 0) -> Array:
    """计算 SPEA2 适应度 (针对特定约束阶段 processcon)。"""
    n = len(objs)
    if processcon == 0 or g is None:
        cv = np.zeros(n, dtype=float)
    elif processcon > totalcon:
        cv = np.sum(np.maximum(0.0, g), axis=1)
    else:
        con_idx = processcon - 1
        cv = np.maximum(0.0, g[:, con_idx])

    # 1. 计算支配矩阵
    dominate = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if cv[i] < cv[j]:
                dominate[i, j] = True
            elif cv[i] > cv[j]:
                dominate[j, i] = True
            else:
                less = np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j])
                more = np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i])
                if less:
                    dominate[i, j] = True
                elif more:
                    dominate[j, i] = True

    # 2. 计算 S(i) & R(i)
    s = np.sum(dominate, axis=1)
    r = np.zeros(n, dtype=float)
    for i in range(n):
        r[i] = np.sum(s[dominate[:, i]])

    # 3. 计算拥挤距离 D(i)
    dist = np.linalg.norm(objs[:, None, :] - objs[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    dist_sort = np.sort(dist, axis=1)
    k_idx = max(0, int(math.floor(math.sqrt(n))) - 1)
    d = 1.0 / (dist_sort[:, k_idx] + 2.0)

    return r + d
