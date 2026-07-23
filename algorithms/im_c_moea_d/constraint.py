"""IM-C-MOEA-D 约束处理模块 (对应 applyConstraintHandling.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array, Population


def apply_constraint_handling(pop: Population) -> Array:
    """约束惩罚转换目标函数值 (对应 applyConstraintHandling.m)。"""
    objs = pop.f.copy()
    cv = pop.cv
    if np.all(cv <= 1e-12):
        return objs

    infeas_mask = cv > 1e-12
    if np.any(infeas_mask):
        f_max = np.max(objs, axis=0)
        f_min = np.min(objs, axis=0)
        span = f_max - f_min
        phi_max = np.max(cv[infeas_mask]) + 1e-15
        objs[infeas_mask] = objs[infeas_mask] + (cv[infeas_mask, None] / phi_max) * span

    return objs
