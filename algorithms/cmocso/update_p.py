"""CMOCSO 种群更新算子模块 (对应 UpdateP.m, UpdateP1.m, UpdateP2.m)。"""
from __future__ import annotations

import numpy as np
from algorithms.cmocso.fitness import cal_fitness
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


def update_p(pop: Population, target_n: int) -> Population:
    """更新可行非支配种群 (对应 UpdateP.m)。"""
    feasible_mask = pop.cv <= 1e-12
    pop_feas = Population(
        x=pop.x[feasible_mask],
        f=pop.f[feasible_mask],
        cv=pop.cv[feasible_mask],
        g=pop.g[feasible_mask] if pop.g is not None else None,
        h=pop.h[feasible_mask] if pop.h is not None else None,
    )
    if len(pop_feas.x) == 0:
        cv_order = np.argsort(pop.cv)[:target_n]
        return Population(
            x=pop.x[cv_order],
            f=pop.f[cv_order],
            cv=pop.cv[cv_order],
            g=pop.g[cv_order] if pop.g is not None else None,
            h=pop.h[cv_order] if pop.h is not None else None,
        )
    if len(pop_feas.x) <= target_n:
        return pop_feas

    fit = cal_fitness(pop_feas.f, pop_feas.cv, var_epsilon=0.0)
    next_mask = fit < 1.0
    if np.sum(next_mask) > target_n:
        next_indices = np.flatnonzero(next_mask)
        del_mask = truncation(pop_feas.f[next_mask], int(np.sum(next_mask) - target_n))
        next_mask[next_indices[del_mask]] = False
    elif np.sum(next_mask) < target_n:
        rank = np.argsort(fit)
        next_mask[rank[:target_n]] = True

    sel_idx = np.flatnonzero(next_mask)
    return Population(
        x=pop_feas.x[sel_idx],
        f=pop_feas.f[sel_idx],
        cv=pop_feas.cv[sel_idx],
        g=pop_feas.g[sel_idx] if pop_feas.g is not None else None,
        h=pop_feas.h[sel_idx] if pop_feas.h is not None else None,
    )


def update_p1(pop: Population, target_n: int, var_epsilon: float) -> tuple[Population, Array]:
    """更新 epsilon 约束协同种群 (对应 UpdateP1.m)。"""
    fit = cal_fitness(pop.f, pop.cv, var_epsilon=var_epsilon)
    next_mask = fit < 1.0
    if np.sum(next_mask) < target_n:
        rank = np.argsort(fit)
        next_mask[rank[:target_n]] = True
    elif np.sum(next_mask) > target_n:
        next_indices = np.flatnonzero(next_mask)
        del_mask = truncation(pop.f[next_mask], int(np.sum(next_mask) - target_n))
        next_mask[next_indices[del_mask]] = False

    sel_idx = np.flatnonzero(next_mask)
    selected_pop = Population(
        x=pop.x[sel_idx],
        f=pop.f[sel_idx],
        cv=pop.cv[sel_idx],
        g=pop.g[sel_idx] if pop.g is not None else None,
        h=pop.h[sel_idx] if pop.h is not None else None,
    )
    return selected_pop, fit[sel_idx]


def update_p2(pop: Population, target_n: int) -> tuple[Population, Array]:
    """更新无约束协同种群 (对应 UpdateP2.m)。"""
    fit = cal_fitness(pop.f, np.zeros_like(pop.cv), var_epsilon=0.0)
    next_mask = fit < 1.0
    if np.sum(next_mask) < target_n:
        rank = np.argsort(fit)
        next_mask[rank[:target_n]] = True
    elif np.sum(next_mask) > target_n:
        next_indices = np.flatnonzero(next_mask)
        del_mask = truncation(pop.f[next_mask], int(np.sum(next_mask) - target_n))
        next_mask[next_indices[del_mask]] = False

    sel_idx = np.flatnonzero(next_mask)
    selected_pop = Population(
        x=pop.x[sel_idx],
        f=pop.f[sel_idx],
        cv=pop.cv[sel_idx],
        g=pop.g[sel_idx] if pop.g is not None else None,
        h=pop.h[sel_idx] if pop.h is not None else None,
    )
    return selected_pop, fit[sel_idx]
