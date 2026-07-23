"""DRLOS-EMCMO 环境选择模块 (对应 EnvironmentalSelection.m)。"""
from __future__ import annotations

import numpy as np
from algorithms.drlos_emcmo.fitness import cal_fitness
from core.schema import Array, Population


def truncation(objs: Array, k: int) -> np.ndarray:
    """基于距离图截断。"""
    n = len(objs)
    dist = np.linalg.norm(objs[:, None, :] - objs[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    del_mask = np.zeros(n, dtype=bool)

    while np.sum(del_mask) < k:
        remain = np.flatnonzero(~del_mask)
        sub_dist = np.sort(dist[np.ix_(remain, remain)], axis=1)
        sorted_indices = np.lexsort(sub_dist.T[::-1])
        del_mask[remain[sorted_indices[0]]] = True

    return del_mask


def environmental_selection(pop: Population, target_n: int, mode: int) -> tuple[Population, Array, np.ndarray]:
    """环境选择 (mode=1 考虑约束，mode=2 不考虑约束)。"""
    cv_input = pop.cv if mode == 1 else None
    fit = cal_fitness(pop.f, cv_input)

    next_mask = fit < 1.0
    if np.sum(next_mask) < target_n:
        rank = np.argsort(fit)
        next_mask[rank[:target_n]] = True
    elif np.sum(next_mask) > target_n:
        next_indices = np.flatnonzero(next_mask)
        del_mask = truncation(pop.f[next_mask], int(np.sum(next_mask) - target_n))
        next_mask[next_indices[del_mask]] = False

    sel_idx = np.flatnonzero(next_mask)
    sub_fit = fit[sel_idx]
    order = np.argsort(sub_fit)
    final_idx = sel_idx[order]

    selected_pop = Population(
        x=pop.x[final_idx],
        f=pop.f[final_idx],
        cv=pop.cv[final_idx],
        g=pop.g[final_idx] if pop.g is not None else None,
        h=pop.h[final_idx] if pop.h is not None else None,
    )
    return selected_pop, sub_fit[order], next_mask
