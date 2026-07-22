"""APSEA 算法的适应度计算模块 (对应 CalFitness.m)。"""
from __future__ import annotations

import math
import numpy as np
from core.schema import Array


def cal_fitness(pop_obj: Array, pop_con: Array | None = None) -> Array:
    """计算每个解的适应度 (对应 PlatEMO 的 CalFitness.m)。

    Args:
        pop_obj: 目标函数值矩阵 (N, M)。
        pop_con: 约束矩阵 (N, K) 或 None。

    Returns:
        Fitness 一维数组 (N,)。适应度越小越优。
    """
    n = len(pop_obj)
    if pop_con is None or pop_con.size == 0:
        cv = np.zeros(n, dtype=float)
    else:
        cv = np.maximum(pop_con, 0.0).sum(axis=1)

    # 1. 检测支配关系
    dominate = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if cv[i] < cv[j]:
                dominate[i, j] = True
            elif cv[i] > cv[j]:
                dominate[j, i] = True
            else:
                # 处于相同约束违反水平时按目标支配关系比较
                k1 = np.any(pop_obj[i] < pop_obj[j])
                k2 = np.any(pop_obj[i] > pop_obj[j])
                if k1 and not k2:  # i 严格支配 j
                    dominate[i, j] = True
                elif k2 and not k1:  # j 严格支配 i
                    dominate[j, i] = True

    # 2. 计算 S(i): 支配其他解的数量
    s = dominate.sum(axis=1)

    # 3. 计算 R(i): 被支配解所接收到的 S(i) 累加强度
    r = (dominate.T * s).sum(axis=1)

    # 4. 计算 D(i): 基于 sqrt(N) 最近邻密度的拥挤度距离
    distances = np.linalg.norm(pop_obj[:, None, :] - pop_obj[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    distances.sort(axis=1)
    
    k_nn = int(math.floor(math.sqrt(n)))
    k_nn = max(0, min(k_nn - 1, n - 2))  # 下标从 0 开始
    
    d = 1.0 / (distances[:, k_nn] + 2.0)

    # 5. 最终 Fitness = R + D
    return r + d
