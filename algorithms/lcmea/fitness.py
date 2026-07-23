"""LCMEA 适应度计算模块 (对应 AdaFitness.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array


def ada_fitness(objs: Array, cv: Array) -> Array:
    """自适应规则下计算个体综合评价值 (对应 AdaFitness.m)。"""
    n = len(objs)
    feas_mask = cv <= 1e-12
    fit = np.zeros(n, dtype=float)

    if np.any(feas_mask):
        fit[feas_mask] = np.sum(objs[feas_mask], axis=1)
    if np.any(~feas_mask):
        fit[~feas_mask] = np.max(objs) + cv[~feas_mask]

    return fit
