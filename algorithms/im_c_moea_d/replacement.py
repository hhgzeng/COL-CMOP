"""IM-C-MOEA-D 全局替换模块 (对应 globalReplacement.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array, Population


def global_replacement(
    pop: Population,
    offsprings: Population,
    weights: Array,
    b_neighbors: Array,
    t_size: int,
    pop_norm_obj: Array,
    off_norm_obj: Array,
) -> Population:
    """基于 Tchebycheff 标量函数的全局替换 (对应 globalReplacement.m)。"""
    n = len(pop.x)
    n_off = len(offsprings.x)
    if n_off == 0:
        return pop

    new_x = pop.x.copy()
    new_f = pop.f.copy()
    new_cv = pop.cv.copy()
    new_g = pop.g.copy() if pop.g is not None else None
    new_norm = pop_norm_obj.copy()

    for i in range(n_off):
        off_x = offsprings.x[i]
        off_f = offsprings.f[i]
        off_cv = offsprings.cv[i]
        off_g = offsprings.g[i] if offsprings.g is not None else None
        off_norm = off_norm_obj[i]

        for j in range(n):
            w_j = weights[j]
            g_old = np.max(new_norm[j] * w_j)
            g_new = np.max(off_norm * w_j)

            # 约束与目标双重判定
            if off_cv < new_cv[j] or (abs(off_cv - new_cv[j]) <= 1e-12 and g_new < g_old):
                new_x[j] = off_x
                new_f[j] = off_f
                new_cv[j] = off_cv
                new_norm[j] = off_norm
                if new_g is not None and off_g is not None:
                    new_g[j] = off_g

    return Population(x=new_x, f=new_f, cv=new_cv, g=new_g)
