"""POCEA 环境选择与参考向量自适应模块 (对应 RVEASelection.m 与 ReferenceVectorAdaptation.m)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array, Population


def association(pop: Population, vs: Array, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """关联分析：将候选解关联至每个参考向量。"""
    pop_f = pop.f - np.min(pop.f, axis=0)
    dis = np.sum(pop_f**2, axis=1)

    norm_f = np.linalg.norm(pop_f, axis=1, keepdims=True) + 1e-15
    norm_v = np.linalg.norm(vs, axis=1, keepdims=True) + 1e-15
    cos_sim = np.dot(pop_f / norm_f, (vs / norm_v).T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    theta = np.arccos(cos_sim)

    index = np.argsort(theta, axis=0)[: min(k, len(pop.x))]
    return index, theta, dis


def rvea_selection(pop: Population, v: Array, target_n: int, t_ratio: float) -> Population:
    """RVEA 角度惩罚与环境选择 (对应 RVEASelection.m)。"""
    n = len(pop.x)
    m = pop.f.shape[1]
    if n <= target_n:
        return pop

    pop_f = pop.f - np.min(pop.f, axis=0)
    norm_f = np.linalg.norm(pop_f, axis=1, keepdims=True) + 1e-15
    norm_v = np.linalg.norm(v, axis=1, keepdims=True) + 1e-15

    cos_sim = np.dot(pop_f / norm_f, (v / norm_v).T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    theta = np.arccos(cos_sim)

    assoc_v = np.argmin(theta, axis=1)
    min_theta = np.min(theta, axis=1)

    gamma = m * (t_ratio**2)
    apd = norm_f.ravel() * (1.0 + gamma * (min_theta / (np.pi / 2.0)))

    sel_indices: list[int] = []
    for i in range(len(v)):
        members = np.flatnonzero(assoc_v == i)
        if len(members) > 0:
            best_m = members[np.argmin(apd[members])]
            sel_indices.append(int(best_m))

    if len(sel_indices) < target_n:
        remain = list(set(range(n)) - set(sel_indices))
        fill_count = target_n - len(sel_indices)
        extra = np.argsort(apd[remain])[:fill_count]
        sel_indices.extend([remain[e] for e in extra])
    elif len(sel_indices) > target_n:
        sel_indices = list(np.array(sel_indices)[np.argsort(apd[sel_indices])[:target_n]])

    sel_idx = np.array(sel_indices, dtype=int)
    return Population(
        x=pop.x[sel_idx],
        f=pop.f[sel_idx],
        cv=pop.cv[sel_idx],
        g=pop.g[sel_idx] if pop.g is not None else None,
        h=pop.h[sel_idx] if pop.h is not None else None,
    )


def reference_vector_adaptation(objs: Array, v0: Array, vs0: Array) -> tuple[Array, Array]:
    """参考向量自适应缩放 (对应 ReferenceVectorAdaptation.m)。"""
    f_min = np.min(objs, axis=0)
    f_max = np.max(objs, axis=0)
    span = f_max - f_min + 1e-15

    v = v0 * span[None, :]
    vs = vs0 * span[None, :]
    return v, vs
