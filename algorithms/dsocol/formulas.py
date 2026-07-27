"""DSOCOL 论文公式模块 (formulas.py)。

包含 Wang 等人 2026 年论文中的所有核心数学计算公式：
- 公式 (2): 约束违反度 CV
- 公式 (3): ε-支配判据、CDP 判据与 SPEA2 适应度 Fitness
- 公式 (4): CSO 失败者速度与位置更新算子
- 公式 (5): S1 获胜者 SBX 交叉算子
- 公式 (6): S2 获胜者最优引导更新算子
- 公式 (7): ε-约束界限动态更新算子
- 公式 (8): COL 趋势学习方向计算
- 公式 (9): COL 正交补空间子代生成
- 权重向量与极坐标夹角计算
"""

from __future__ import annotations

from math import comb
import numpy as np

from core.schema import Array


# ---------------------------------------------------------------------------
# 公式 (2) & (3): 支配判据与 SPEA2 适应度公式
# ---------------------------------------------------------------------------

def epsilon_dominates(
    fa: Array, cva: float, fb: Array, cvb: float, epsilon: float, atol: float = 1e-12
) -> bool:
    """公式 (3): ε-约束支配规则。"""
    a_relaxed, b_relaxed = cva <= epsilon + atol, cvb <= epsilon + atol
    if a_relaxed and b_relaxed:
        return bool(np.all(fa <= fb + atol) and np.any(fa < fb - atol))
    if a_relaxed != b_relaxed:
        return a_relaxed
    return cva < cvb - atol


def cdp_dominates(fa: Array, cva: float, fb: Array, cvb: float) -> bool:
    """公式 (3): 约束支配原则 (CDP, 即 ε=0)。"""
    return epsilon_dominates(fa, cva, fb, cvb, 0.0)


def spea2_fitness(f: Array, cv: Array, epsilon: float) -> Array:
    """公式 (3): SPEA2 支配强度 R_eps(x) 与最近邻欧氏距离密度 D(x)。"""
    n = len(f)
    dominates = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                dominates[i, j] = epsilon_dominates(f[i], cv[i], f[j], cv[j], epsilon)

    strength = dominates.sum(axis=1)
    raw = (dominates.T * strength).sum(axis=1)

    distances = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    density = 1.0 / (nearest + 2.0)
    return raw + density


def nondominated_feasible_indices(f: Array, cv: Array) -> Array:
    """提取可行且满足 CDP 非支配关系的个体索引。"""
    feasible = np.flatnonzero(cv <= 1e-12)
    return np.array(
        [
            i
            for i in feasible
            if not any(
                cdp_dominates(f[j], cv[j], f[i], cv[i]) for j in feasible if j != i
            )
        ],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# 权重向量与空间夹角辅助计算
# ---------------------------------------------------------------------------

def simplex_weight_vectors(m: int, requested: int) -> Array:
    """生成均匀单纯形格点权重向量，数量接近 N/10。"""
    h = 1
    while comb(h + m - 1, m - 1) < requested:
        h += 1

    def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
        if parts == 1:
            return [(total,)]
        return [
            (i,) + rest
            for i in range(total + 1)
            for rest in compositions(total - i, parts - 1)
        ]

    return np.asarray(compositions(h, m), dtype=float) / h


def calc_angles(f: Array, weights: Array) -> Array:
    """计算目标空间解向量与均匀权重向量的夹角距离。"""
    shifted = f - np.min(f, axis=0, keepdims=True)
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    unit_f = np.divide(shifted, norms, out=np.zeros_like(shifted), where=norms > 1e-15)
    unit_w = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    return np.arccos(np.clip(unit_f @ unit_w.T, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 公式 (4), (5), (6): CSO 算子公式
# ---------------------------------------------------------------------------

def update_cso_loser(
    x_w: Array, x_l: Array, v_l: Array, rng: np.random.Generator
) -> tuple[Array, Array]:
    """公式 (4): CSO 失败者速度与位置更新。

    V_l = r0 * V_l + r1 * (X_w - X_l)
    X_l = X_l + V_l
    """
    v_new = rng.random((len(x_l), 1)) * v_l + rng.random((len(x_l), 1)) * (x_w - x_l)
    x_new = x_l + v_new
    return x_new, v_new


def update_cso_winner_s1(
    x_w: Array, rng: np.random.Generator, distribution_index: float = 20.0
) -> Array:
    """公式 (5): S1 获胜者基于中心点与配对差值的 SBX 交叉更新。"""
    x_new = x_w.copy()
    w_indices = rng.permutation(len(x_w))
    half = len(w_indices) // 2
    left, right = w_indices[:half], w_indices[half:]
    center, delta = (x_new[left] + x_new[right]) / 2.0, x_new[left] - x_new[right]
    u = rng.random(delta.shape)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (distribution_index + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (distribution_index + 1.0)),
    )
    x_new[left] = center + beta * delta / 2.0
    x_new[right] = center - beta * delta / 2.0
    return x_new


def update_cso_winner_s2(
    x_w: Array, fitness_w: Array, rng: np.random.Generator
) -> Array:
    """公式 (6): S2 获胜者向当前适应度最佳解与随机配对差值的引导更新。"""
    x_new = x_w.copy()
    best_idx = np.argmin(fitness_w)
    best_x = x_w[best_idx]
    w_indices = np.arange(len(x_w))

    for k in w_indices:
        choices = w_indices[w_indices != k]
        if len(choices) >= 2:
            i, j = rng.choice(choices, size=2, replace=False)
        elif len(choices) == 1:
            i = j = choices[0]
        else:
            i = j = k
        x_new[k] += (best_x - x_new[k]) / 2.0 + (x_w[i] - x_w[j]) / 2.0

    return x_new


# ---------------------------------------------------------------------------
# 公式 (8) & (9): COL 趋势学习与正交补学习算子公式
# ---------------------------------------------------------------------------

def compute_col_trend_direction(
    winner_x: Array, loser_x: Array, boundary_x: Array, tau: float
) -> Array:
    """公式 (8): 计算 COL 趋势学习收敛方向单位向量 v。

    direction = tau * (x_w - x_l) + (1 - tau) * (x_w - x_r)
    """
    d1 = winner_x - loser_x
    d2 = winner_x - boundary_x
    direction = tau * d1 + (1.0 - tau) * d2
    norm = np.linalg.norm(direction)
    if norm > 1e-15:
        direction /= norm
    return direction


def compute_col_orthogonal_direction(direction: Array, rng: np.random.Generator) -> Array:
    """公式 (9): 将随机向量投影到 direction 的正交补空间，生成单位正交方向。"""
    random_v = rng.normal(size=direction.shape)
    perpendicular = random_v - random_v.dot(direction) * direction
    pn = np.linalg.norm(perpendicular)
    if pn > 1e-15:
        perpendicular /= pn
    return perpendicular


# ---------------------------------------------------------------------------
# 公式 (7): ε-约束放松下限与动态更新公式
# ---------------------------------------------------------------------------

def update_epsilon(
    epsilon: float,
    ratio: float,
    alpha: float,
    t_max: float,
    epsilon_max: float,
    rng: np.random.Generator,
) -> float:
    """论文公式 (7): 根据主群体可行率 ratio 动态更新 ε 边界。

    ε(t+1) = (1 - σ) * ε(t), if fr <= α
    ε(t+1) = ε_max, otherwise
    """
    if ratio <= alpha:
        power = np.power(1.0 / max(epsilon_max, 1e-15), t_max / 3.0)
        sigma_min = float(np.clip(1.0 - power, 0.0, 0.95))
        sigma = rng.uniform(sigma_min, 1.0)
        return float((1.0 - sigma) * epsilon)
    return float(epsilon_max)
