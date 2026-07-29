"""IM-C-MOEA-D 算子与聚类模块 (对应 Operator.m)。"""
from __future__ import annotations

import numpy as np
from core.operators import polynomial_mutation
from core.schema import CMOP, Array, Population


def kmeans_clusters(objs: Array, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-Means 算法归类，返回样本所属类别标签 (N,)。"""
    n = len(objs)
    if n <= k:
        return np.arange(n)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = objs[init_idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(15):
        dist = np.linalg.norm(objs[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(k):
            members = objs[labels == i]
            new_centers[i] = np.mean(members, axis=0) if len(members) > 0 else centers[i]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels


def gp_operator(
    problem: CMOP,
    pop: Population,
    pop_objs: Array,
    rng: np.random.Generator,
    l_dim: int = 3,
) -> Population:
    """基于高斯过程 (GP) 逆向建模的生殖算子 (对应 Operator.m)。"""
    pop_dec = pop.x
    n, d = pop_dec.shape
    m = problem.n_obj
    if d < 3:
        l_dim = d

    if n < 2 * m:
        off_dec = pop_dec.copy()
    else:
        off_dec_list = []
        fmin = 1.5 * np.min(pop_objs, axis=0) - 0.5 * np.max(pop_objs, axis=0)
        fmax = 1.5 * np.max(pop_objs, axis=0) - 0.5 * np.min(pop_objs, axis=0)
        n_sub = n // m

        for m_idx in range(m):
            parents = rng.choice(n, size=n_sub, replace=False)
            off_sub = pop_dec[parents].copy()
            selected_dims = rng.choice(d, size=min(l_dim, d), replace=False)
            x_test = np.linspace(fmin[m_idx], fmax[m_idx], len(parents))[:, None]

            for d_idx in selected_dims:
                try:
                    # 线性核 GP Regression (@covLIN): K = X X^T + s2 I
                    x_tr = pop_objs[parents, m_idx:m_idx+1]  # (K, 1)
                    y_tr = pop_dec[parents, d_idx:d_idx+1]          # (K, 1)
                    s2 = 1e-4

                    k_xx = x_tr @ x_tr.T + s2 * np.eye(len(parents))
                    k_star_x = x_test @ x_tr.T
                    k_star_star = x_test @ x_test.T + s2 * np.eye(len(x_test))

                    alpha_vec = np.linalg.solve(k_xx, y_tr)
                    ymu = (k_star_x @ alpha_vec).flatten()

                    v_mat = np.linalg.solve(k_xx, k_star_x.T)
                    ys2_diag = np.diag(k_star_star - k_star_x @ v_mat)
                    ys2_diag = np.maximum(1e-12, ys2_diag)

                    sample = ymu + rng.random() * np.sqrt(ys2_diag) * rng.standard_normal(len(ys2_diag))
                    off_sub[:, d_idx] = sample
                except Exception:
                    pass

            off_dec_list.append(off_sub)

        off_dec = np.vstack(off_dec_list)

    # 将超出边界的值替换为均匀随机值
    n_off, d_off = off_dec.shape
    lo_rep = np.tile(problem.lower, (n_off, 1))
    hi_rep = np.tile(problem.upper, (n_off, 1))
    rand_dec = rng.uniform(lo_rep, hi_rep)
    invalid = (off_dec < lo_rep) | (off_dec > hi_rep)
    off_dec[invalid] = rand_dec[invalid]

    # 多项式变异
    off_dec = polynomial_mutation(off_dec, problem.lower, problem.upper, rng)
    res = problem.evaluate(off_dec)
    return Population(x=off_dec, f=res.f, cv=res.cv, g=res.g, h=res.h)
