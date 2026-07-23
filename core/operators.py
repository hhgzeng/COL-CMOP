"""公共遗传算子模块，包含二元锦标赛选择、SBX 交叉与多项式变异 (OperatorGA)。"""
from __future__ import annotations

import numpy as np
from core.schema import Array


def tournament_selection(k: int, n_select: int, fitness: Array, rng: np.random.Generator | None = None) -> Array:
    """k-元锦标赛选择，根据适应度 Fitness (越小越优) 选择 n_select 个样本索引。

    Args:
        k: 锦标赛规模 (如 2)。
        n_select: 需要选出的索引数量。
        fitness: 适应度数组 (N,)。
        rng: 随机数生成器。

    Returns:
        选中的索引数组 (n_select,)。
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(fitness)
    selected = np.empty(n_select, dtype=int)
    replace_choice = n < k
    for i in range(n_select):
        competitors = rng.choice(n, size=k, replace=replace_choice)
        selected[i] = competitors[np.argmin(fitness[competitors])]
    return selected



def operator_ga(
    x: Array,
    lower: Array,
    upper: Array,
    rng: np.random.Generator | None = None,
    pc: float = 1.0,
    pm: float | None = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
) -> Array:
    """通用二进制交叉 (SBX) + 多项式变异 (PM) 算子 (对应 PlatEMO 的 OperatorGA)。

    Args:
        x: 父代决策变量矩阵 (N, D)。
        lower: 变量下界 (D,)。
        upper: 变量上界 (D,)。
        rng: 随机数生成器。
        pc: 交叉概率。
        pm: 变异概率 (默认 1/D)。
        eta_c: SBX 交叉分布指数。
        eta_m: PM 变异分布指数。

    Returns:
        生成的子代决策变量矩阵 (N, D)。
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = x.shape
    if pm is None:
        pm = 1.0 / d

    lower_2d = lower[None, :]
    upper_2d = upper[None, :]

    # 1. SBX 交叉 (两两配对)
    order = rng.permutation(n)
    offspring_x = x[order].copy()

    for i in range(0, n - 1, 2):
        p1, p2 = order[i], order[i + 1]
        if rng.random() <= pc:
            y1 = x[p1].copy()
            y2 = x[p2].copy()

            u = rng.random(d)
            beta = np.where(
                u <= 0.5,
                (2.0 * u) ** (1.0 / (eta_c + 1.0)),
                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
            )

            c1 = 0.5 * ((1.0 + beta) * y1 + (1.0 - beta) * y2)
            c2 = 0.5 * ((1.0 - beta) * y1 + (1.0 + beta) * y2)

            offspring_x[i] = c1
            offspring_x[i + 1] = c2

    # 2. 多项式变异 (PM)
    mask = rng.random((n, d)) < pm
    if np.any(mask):
        u = rng.random((n, d))
        delta1 = (offspring_x - lower_2d) / (upper_2d - lower_2d + 1e-15)
        delta2 = (upper_2d - offspring_x) / (upper_2d - lower_2d + 1e-15)

        val = 2.0 * u + (1.0 - 2.0 * u) * np.power(np.maximum(0.0, 1.0 - delta1), eta_m + 1.0)
        val = np.clip(val, 0.0, None)

        delta_q = np.where(
            u <= 0.5,
            np.power(val, 1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - np.power(np.maximum(0.0, 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(np.maximum(0.0, 1.0 - delta2), eta_m + 1.0)), 1.0 / (eta_m + 1.0)),
        )

        span = np.broadcast_to(upper_2d - lower_2d, offspring_x.shape)
        offspring_x[mask] = offspring_x[mask] + (delta_q * span)[mask]

    return np.clip(offspring_x, lower_2d, upper_2d)


def polynomial_mutation(
    x: Array,
    lower: Array,
    upper: Array,
    rng: np.random.Generator | None = None,
    eta_m: float = 20.0,
    pm: float | None = None,
) -> Array:
    """标准多项式变异 (Polynomial Mutation)。"""
    if rng is None:
        rng = np.random.default_rng()

    n, d = x.shape
    if pm is None:
        pm = 1.0 / d

    mask = rng.random((n, d)) < pm
    if not np.any(mask):
        return x

    x_mut = x.copy()
    u = rng.random((n, d))
    lower_2d = lower[None, :]
    upper_2d = upper[None, :]

    delta1 = (x - lower_2d) / (upper_2d - lower_2d + 1e-15)
    delta2 = (upper_2d - x) / (upper_2d - lower_2d + 1e-15)

    val = 2.0 * u + (1.0 - 2.0 * u) * np.power(np.maximum(0.0, 1.0 - delta1), eta_m + 1.0)
    val = np.clip(val, 0.0, None)

    delta_q = np.where(
        u <= 0.5,
        np.power(val, 1.0 / (eta_m + 1.0)) - 1.0,
        1.0 - np.power(np.maximum(0.0, 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(np.maximum(0.0, 1.0 - delta2), eta_m + 1.0)), 1.0 / (eta_m + 1.0)),
    )

    span = np.broadcast_to(upper_2d - lower_2d, x.shape)
    x_mut[mask] = x[mask] + (delta_q * span)[mask]
    return np.clip(x_mut, lower_2d, upper_2d)


def operator_de(
    x: Array,
    p1: Array,
    p2: Array,
    lower: Array,
    upper: Array,
    rng: np.random.Generator | None = None,
    CR: float = 1.0,
    F: float = 0.5,
    eta_m: float = 20.0,
    pm: float | None = None,
) -> Array:
    """通用 Differential Evolution (DE/rand/1/bin) 算子 + 多项式变异。"""
    if rng is None:
        rng = np.random.default_rng()

    n, d = x.shape
    if pm is None:
        pm = 1.0 / d

    lower_2d = lower[None, :]
    upper_2d = upper[None, :]

    mask = rng.random((n, d)) < CR
    j_rand = rng.integers(0, d, size=n)
    for i in range(n):
        mask[i, j_rand[i]] = True

    v = x + F * (p1 - p2)
    offspring_x = np.where(mask, v, x)
    offspring_x = np.clip(offspring_x, lower_2d, upper_2d)

    return polynomial_mutation(offspring_x, lower, upper, rng, eta_m=eta_m, pm=pm)


def uniform_point(n: int, m: int) -> tuple[Array, int]:
    """生成 M 维单纯形上约 N 个均匀分布的参考点/权重向量 (Das & Dennis 方法)。"""
    import math

    if m == 1:
        return np.ones((1, 1), dtype=float), 1

    H = 1
    while math.comb(H + m - 1, m - 1) <= n:
        H += 1
    H = max(1, H - 1)

    def _generate_recursive(m_rem: int, sum_rem: int):
        if m_rem == 1:
            yield (sum_rem,)
        else:
            for i in range(sum_rem + 1):
                for sub in _generate_recursive(m_rem - 1, sum_rem - i):
                    yield (i,) + sub

    w = np.array(list(_generate_recursive(m, H)), dtype=float) / float(H)
    return w, len(w)


