"""IM-C-MOEA-D 约束处理模块 (对应 applyConstraintHandling.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array, Population


def apply_constraint_handling(pop: Population) -> Array:
    """约束惩罚转换目标函数值 (对应 applyConstraintHandling.m)。

    Infeasible 个体在每个目标维度上都加上 max(PopObj, [], 1) + cv。
    """
    objs = pop.f.copy()
    cv = pop.cv
    infeasible = cv > 1e-12
    if np.any(infeasible):
        max_objs = np.max(objs, axis=0)
        objs[infeasible] = max_objs + cv[infeasible, None]
    return objs
