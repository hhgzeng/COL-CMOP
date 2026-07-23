"""C3M 归档管理模块 (对应 archive.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Population


def archive_selection(pop: Population, target_n: int) -> Population:
    """归档集更新 (可行 + NSGA-II 拥挤度筛选)。"""
    feasible_mask = pop.cv <= 1e-12
    if not np.any(feasible_mask):
        return Population(x=np.empty((0, pop.x.shape[1])), f=np.empty((0, pop.f.shape[1])), cv=np.empty(0))

    feas_idx = np.flatnonzero(feasible_mask)
    feas_f = pop.f[feas_idx]

    # 非支配排序
    n_f = len(feas_f)
    dom = np.zeros((n_f, n_f), dtype=bool)
    for i in range(n_f):
        for j in range(n_f):
            if i != j and np.all(feas_f[i] <= feas_f[j]) and np.any(feas_f[i] < feas_f[j]):
                dom[i, j] = True

    rank1_mask = np.sum(dom, axis=0) == 0
    rank1_idx = feas_idx[rank1_mask]

    if len(rank1_idx) <= target_n:
        sel_idx = rank1_idx
    else:
        # 计算拥挤距离
        f_sub = pop.f[rank1_idx]
        n_sub, m = f_sub.shape
        crowd_dis = np.zeros(n_sub, dtype=float)
        f_max = np.max(f_sub, axis=0)
        f_min = np.min(f_sub, axis=0)

        for i in range(m):
            order = np.argsort(f_sub[:, i])
            crowd_dis[order[0]] = np.inf
            crowd_dis[order[-1]] = np.inf
            span = f_max[i] - f_min[i] + 1e-15
            for j in range(1, n_sub - 1):
                crowd_dis[order[j]] += (f_sub[order[j + 1], i] - f_sub[order[j - 1], i]) / span

        rank = np.argsort(-crowd_dis)
        sel_idx = rank1_idx[rank[:target_n]]

    return Population(
        x=pop.x[sel_idx],
        f=pop.f[sel_idx],
        cv=pop.cv[sel_idx],
        g=pop.g[sel_idx] if pop.g is not None else None,
        h=pop.h[sel_idx] if pop.h is not None else None,
    )
