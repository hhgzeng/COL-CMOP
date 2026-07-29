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
    pop_penalized_obj: Array,
    off_penalized_obj: Array,
    rng: np.random.Generator,
) -> Population:
    """基于 Tchebycheff 标量函数的全局/邻域替换 (对应 globalReplacement.m)。"""
    n_off = len(offsprings.x)
    if n_off == 0:
        return pop

    new_x = pop.x.copy()
    new_f = pop.f.copy()
    new_cv = pop.cv.copy()
    new_g = pop.g.copy() if pop.g is not None else None
    new_h = pop.h.copy() if pop.h is not None else None
    new_penalized = pop_penalized_obj.copy()

    for i in range(n_off):
        off_pen = off_penalized_obj[i]
        off_cv = offsprings.cv[i]

        # 1. 计算每个权重向量对应的切比雪夫值
        tch_vals = np.max(off_pen * weights, axis=1)
        best_weight_idx = int(np.argmin(tch_vals))

        # 2. 随机打乱最佳权重向量的 T 邻域下标
        neighbors = b_neighbors[best_weight_idx].copy()
        rng.shuffle(neighbors)
        P = neighbors

        # 3. 计算旧邻居与新个体的切比雪夫值
        g_old = np.max(new_penalized[P] * weights[P], axis=1)
        g_new = np.max(off_pen[None, :] * weights[P], axis=1)

        cvo = off_cv
        cvp = new_cv[P]

        # 4. 寻找满足 g_old >= g_new 且 CVP >= CVO 的邻居个体，至多替换 T 个
        replace_mask = (g_old >= g_new) & (cvp >= cvo)
        replace_indices = P[replace_mask][:t_size]

        if len(replace_indices) > 0:
            new_x[replace_indices] = offsprings.x[i]
            new_f[replace_indices] = offsprings.f[i]
            new_cv[replace_indices] = offsprings.cv[i]
            new_penalized[replace_indices] = off_pen
            if new_g is not None and offsprings.g is not None:
                new_g[replace_indices] = offsprings.g[i]
            if new_h is not None and offsprings.h is not None:
                new_h[replace_indices] = offsprings.h[i]

    return Population(x=new_x, f=new_f, cv=new_cv, g=new_g, h=new_h)
