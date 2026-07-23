"""LCMEA 环境选择模块 (对应 EnvCDP.m, EnvEpsilon.m, EnvRL.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Population


def env_cdp_selection(pop: Population, target_n: int) -> Population:
    """基于 CDP 规则的选择 (对应 EnvCDP.m)。"""
    cv = pop.cv
    feas_mask = cv <= 1e-12

    if np.sum(feas_mask) >= target_n:
        feas_idx = np.flatnonzero(feas_mask)
        f_sub = pop.f[feas_idx]
        order = np.argsort(f_sub[:, 0])[:target_n]
        sel_idx = feas_idx[order]
    else:
        cv_order = np.argsort(cv)
        sel_idx = cv_order[:target_n]

    return Population(
        x=pop.x[sel_idx],
        f=pop.f[sel_idx],
        cv=pop.cv[sel_idx],
        g=pop.g[sel_idx] if pop.g is not None else None,
        h=pop.h[sel_idx] if pop.h is not None else None,
    )


def env_epsilon_selection(pop: Population, target_n: int, var_epsilon: float) -> Population:
    """基于 Epsilon 约束的选择 (对应 EnvEpsilon.m)。"""
    eps_cv = np.maximum(0.0, pop.cv - var_epsilon)
    order = np.argsort(eps_cv + np.sum(pop.f, axis=1))[:target_n]
    return Population(
        x=pop.x[order],
        f=pop.f[order],
        cv=pop.cv[order],
        g=pop.g[order] if pop.g is not None else None,
        h=pop.h[order] if pop.h is not None else None,
    )
