"""DSOCOL 算法伪代码流程模块 (algorithms.py)。

包含 Wang 等人 2026 年论文中的全部 Algorithm 1--5 算法流程：
- Algorithm 1: The framework of DSOCOL (DSOCOL 主框架)
- Algorithm 2: Offspring Generation (子代生成流程)
- Algorithm 3: Collaborative Orthogonal Learning (协同正交学习流程)
- Algorithm 4: Environmental Selection (基于 SPEA2 截断的环境选择)
- Algorithm 5: Niche-Guided Subset Selection (生态位引导子集选择 NGSS)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.operators import polynomial_mutation
from core.schema import CMOP, Array, Population, Result
from algorithms.dsocol.formulas import (
    calc_angles,
    compute_col_orthogonal_direction,
    compute_col_trend_direction,
    nondominated_feasible_indices,
    simplex_weight_vectors,
    spea2_fitness,
    update_cso_loser,
    update_cso_winner_s1,
    update_cso_winner_s2,
    update_epsilon,
)


def take_pop(pop: Population, idx: Array) -> Population:
    """按下标提取 Population 子集。"""
    return Population(
        x=pop.x[idx],
        f=pop.f[idx],
        cv=pop.cv[idx],
        g=pop.g[idx] if pop.g is not None else None,
        h=pop.h[idx] if pop.h is not None else None,
    )


def merge_pops(*pops: Population) -> Population:
    """合并多个 Population。"""
    x_concat = np.concatenate([p.x for p in pops], axis=0)
    f_concat = np.concatenate([p.f for p in pops], axis=0)
    cv_concat = np.concatenate([p.cv for p in pops], axis=0)
    g_concat = np.concatenate([p.g for p in pops if p.g is not None], axis=0) if pops[0].g is not None else None
    h_concat = np.concatenate([p.h for p in pops if p.h is not None], axis=0) if pops[0].h is not None else None
    return Population(x=x_concat, f=f_concat, cv=cv_concat, g=g_concat, h=h_concat)


# ===========================================================================
# Algorithm 2: Offspring Generation (子代生成流程)
# ===========================================================================

def algorithm2_offspring_generation(
    problem: CMOP,
    pop: Population,
    velocity: Array,
    fitness: Array,
    swarm_type: int,
    rng: np.random.Generator,
    distribution_index: float = 20.0,
) -> tuple[Population, Array]:
    """Algorithm 2: Offspring Generation (子代生成)。

    1. 两两随机竞争，分为获胜组 Sw 与失败组 Sl
    2. 调用公式 (4) 更新失败组 Sl 的速度与位置
    3. 群体 1 调用公式 (5) 对 Sw 进行 SBX 交叉；群体 2 调用公式 (6) 最佳引导 Sw
    4. 执行多项式变异与边界裁剪，评估返回新子代与速度
    """
    n = len(pop.x)
    order = rng.permutation(n)
    winners: list[int] = []
    losers: list[int] = []
    for a, b in order.reshape(-1, 2):
        if fitness[a] < fitness[b]:
            winners.append(int(a))
            losers.append(int(b))
        else:
            winners.append(int(b))
            losers.append(int(a))

    w = np.asarray(winners, dtype=int)
    l = np.asarray(losers, dtype=int)  # noqa: E741
    x, v = pop.x.copy(), velocity.copy()

    # 调用公式 (4)
    x[l], v[l] = update_cso_loser(x[w], x[l], v[l], rng)

    if swarm_type == 1:
        # 调用公式 (5)
        x[w] = update_cso_winner_s1(x[w], rng, distribution_index)
    else:
        # 调用公式 (6)
        x[w] = update_cso_winner_s2(x[w], fitness[w], rng)

    x = polynomial_mutation(x, problem.lower, problem.upper, rng)
    x = np.clip(x, problem.lower, problem.upper)

    res = problem.evaluate(x)
    offspring_pop = Population(x=np.asarray(x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)
    return offspring_pop, v


# ===========================================================================
# Algorithm 3: Collaborative Orthogonal Learning (COL 流程)
# ===========================================================================

def algorithm3_collaborative_orthogonal_learning(
    problem: CMOP,
    s1: Population,
    s2: Population,
    weights: Array,
    tau: float,
    rng: np.random.Generator,
) -> tuple[Population, Population]:
    """Algorithm 3: Collaborative Orthogonal Learning (COL 协同正交学习)。

    1. 基于权重向量与角度选出候选解
    2. 调用公式 (8) 趋势学习生成主群体的扩展解 main
    3. 调用公式 (9) 正交补平面对 S2 进行采样生成扩展解 aux
    """
    n = len(s1.x)
    quota = max(1, n // len(weights))
    a1, a2 = calc_angles(s1.f, weights), calc_angles(s2.f, weights)
    main, aux = [], []
    lower, upper = problem.lower, problem.upper

    for j in range(len(weights)):
        # S1 趋势学习
        cand1 = np.argsort(a1[:, j])[:quota]
        nd = cand1[nondominated_feasible_indices(s1.f[cand1], s1.cv[cand1])]
        rem = np.array([i for i in cand1 if i not in nd], dtype=int)
        if len(cand1) == 1:
            winner = loser = cand1[0]
        elif len(nd) >= 2 and len(rem) == 0:
            winner, loser = rng.choice(nd, size=2, replace=False)
        elif len(nd) >= 1 and len(rem) >= 1:
            winner, loser = rng.choice(nd), rng.choice(rem)
        else:
            ranked = cand1[np.argsort(s1.cv[cand1])]
            mid = max(1, len(ranked) // 2)
            winner = rng.choice(ranked[:mid])
            loser = rng.choice(ranked[mid:]) if mid < len(ranked) else winner

        boundary = upper if rng.random() < 0.5 else lower
        # 调用公式 (8)
        direction = compute_col_trend_direction(s1.x[winner], s1.x[loser], boundary, tau)
        main.append(s1.x[winner] + rng.uniform(-1, 1) * direction)

        # S2 正交补学习
        cand2 = np.argsort(a2[:, j])[:quota]
        feasible_nd = cand2[nondominated_feasible_indices(s2.f[cand2], s2.cv[cand2])]
        anchor = (
            rng.choice(feasible_nd)
            if len(feasible_nd)
            else cand2[np.argmin(s2.cv[cand2])]
        )
        # 调用公式 (9)
        perpendicular = compute_col_orthogonal_direction(direction, rng)
        aux.append(s2.x[anchor] + rng.uniform(-1, 1) * perpendicular)

    main_arr = np.clip(np.asarray(main), lower, upper)
    aux_arr = np.clip(np.asarray(aux), lower, upper)

    main_arr = polynomial_mutation(main_arr, lower, upper, rng)
    aux_arr = polynomial_mutation(aux_arr, lower, upper, rng)

    res_main = problem.evaluate(main_arr)
    res_aux = problem.evaluate(aux_arr)

    pop_main = Population(x=main_arr, f=res_main.f, cv=res_main.cv, g=res_main.g, h=res_main.h)
    pop_aux = Population(x=aux_arr, f=res_aux.f, cv=res_aux.cv, g=res_aux.g, h=res_aux.h)
    return pop_main, pop_aux


# ===========================================================================
# Algorithm 4: Environmental Selection (环境选择流程)
# ===========================================================================

def algorithm4_spea2_truncate(f: Array, indices: list[Any], target: int) -> list[int]:
    """Algorithm 4 依赖的 SPEA2 字典序拥挤度截断辅助过程。"""
    indices = indices.copy()
    while len(indices) > target:
        points = f[indices]
        d = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        remove = min(range(len(indices)), key=lambda i: tuple(np.sort(d[i])))
        indices.pop(remove)
    return indices


def algorithm4_environmental_selection(
    pop: Population, vel: Array, target_n: int, epsilon: float
) -> tuple[Population, Array, Array]:
    """Algorithm 4: Environmental Selection (环境选择)。

    1. 计算公式 (3) 的 SPEA2 适应度
    2. 保留 Fit < 1 的非支配集，超出 target_n 时进行 SPEA2 截断
    """
    fit = spea2_fitness(pop.f, pop.cv, epsilon)
    chosen = list(np.flatnonzero(fit < 1.0))
    if len(chosen) < target_n:
        ranked_remainder = [i for i in np.argsort(fit) if i not in chosen]
        chosen += ranked_remainder[: target_n - len(chosen)]
    elif len(chosen) > target_n:
        chosen = algorithm4_spea2_truncate(pop.f, chosen, target_n)

    idx = np.asarray(chosen, dtype=int)
    selected_pop = take_pop(pop, idx)
    selected_vel = (
        vel[idx] if vel is not None and len(vel) == len(pop.x) else np.zeros_like(selected_pop.x)
    )
    return selected_pop, selected_vel, spea2_fitness(selected_pop.f, selected_pop.cv, epsilon)


# ===========================================================================
# Algorithm 5: Niche-Guided Subset Selection (NGSS 生态位选择流程)
# ===========================================================================

def algorithm5_niche_guided_subset_selection(
    pop: Population, vel: Array, target_n: int, weights: Array
) -> tuple[Population, Array, Array]:
    """Algorithm 5: Niche-Guided Subset Selection (NGSS)。

    1. 计算夹角并划分生态位 Niche
    2. 按非支配可行集 -> 内部密度最稀疏剔除 -> 剩余全员聚类填充多层逻辑选择
    """
    k, capacity = len(weights), target_n // len(weights)
    angle = calc_angles(pop.f, weights)
    niche = angle.argmin(axis=1)

    nd = set(nondominated_feasible_indices(pop.f, pop.cv).tolist())
    selected: list[int] = []
    remainder: list[int] = [i for i in range(len(pop.x)) if i not in nd]

    for j in range(k):
        members = [i for i in nd if niche[i] == j]
        if len(members) > capacity:
            d = np.linalg.norm(
                pop.f[members, None] - pop.f[np.array(members)][None], axis=2
            )
            np.fill_diagonal(d, np.inf)
            keep = np.asarray(members)[
                np.argsort(d.min(axis=1))[::-1][:capacity]
            ].tolist()
            remainder += [i for i in members if i not in keep]
            members = keep
        selected += members

    for j in range(k):
        need = capacity - sum(niche[i] == j for i in selected)
        pool = [i for i in remainder if niche[i] == j and i not in selected]
        if need > 0:
            if len(pool) > need:
                pool.sort(key=lambda i: (pop.cv[i] > 1e-12, pop.cv[i], angle[i, j]))
            selected += pool[:need]

    if len(selected) < target_n:
        remaining = [i for i in range(len(pop.x)) if i not in selected]
        remaining.sort(key=lambda i: (pop.cv[i] > 1e-12, pop.cv[i], angle[i].min()))
        selected += remaining[: target_n - len(selected)]

    idx = np.asarray(selected[:target_n], dtype=int)
    out_pop = take_pop(pop, idx)
    selected_vel = (
        vel[idx] if vel is not None and len(vel) == len(pop.x) else np.zeros_like(out_pop.x)
    )
    return out_pop, selected_vel, spea2_fitness(out_pop.f, out_pop.cv, 0.0)


# ===========================================================================
# Algorithm 1: The Framework of DSOCOL (DSOCOL 主框架)
# ===========================================================================

class DSOCOL:
    """Algorithm 1: The framework of DSOCOL."""

    def __init__(
        self,
        population_size: int = 100,
        col_frequency: int = 75,
        alpha: float = 0.95,
        distribution_index: float = 20.0,
        seed: int | None = None,
    ):
        if population_size < 4 or population_size % 2:
            raise ValueError("种群规模必须是大于等于 4 的偶数")
        self.n = population_size
        self.col_frequency = col_frequency
        self.alpha = alpha
        self.distribution_index = distribution_index
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, problem: CMOP, x: np.ndarray) -> Population:
        res = problem.evaluate(x)
        return Population(x=np.asarray(x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)

    def run(self, problem: CMOP) -> Result:
        """基于 FE 预算 (problem.max_evals) 执行 Algorithm 1。"""
        lo, hi = problem.lower, problem.upper

        # Line 1: 生成初始双群体 S1 和 S2 及其速度矩阵
        x1 = self.rng.uniform(lo, hi, size=(self.n, len(lo)))
        x2 = self.rng.uniform(lo, hi, size=(self.n, len(lo)))
        v1 = np.zeros_like(x1)
        v2 = np.zeros_like(x2)

        s1 = self._evaluate(problem, x1)
        s2 = self._evaluate(problem, x2)

        # Line 2-4: 初始化权重向量 W 与控制参数
        weights = simplex_weight_vectors(s1.f.shape[1], max(1, self.n // 10))
        epsilon_max = max(float(s1.cv.max()), 1e-15)
        epsilon = epsilon_max

        # Line 5-6: 评估适应度 F1, F2
        f1 = spea2_fitness(s1.f, s1.cv, epsilon)
        f2 = spea2_fitness(s2.f, s2.cv, 0.0)

        history: dict[str, list[float]] = {
            "fe": [],
            "epsilon": [],
            "feasible_ratio_s1": [],
            "feasible_ratio_s2": [],
        }

        t = 0
        total_generations = max(1, problem.max_evals // (2 * self.n))
        tc = max(1, total_generations // 3)  # noqa

        # Line 7: while t <= T (在此通过 problem.eval_count < problem.max_evals 精确驱动)
        while problem.eval_count < problem.max_evals:
            t += 1

            v1_old, v2_old = v1.copy(), v2.copy()

            # Line 8-9: 调用 Algorithm 2 生成子代 O1, O2
            o1, v1 = algorithm2_offspring_generation(
                problem, s1, v1, f1, 1, self.rng, self.distribution_index
            )
            o2, v2 = algorithm2_offspring_generation(
                problem, s2, v2, f2, 2, self.rng, self.distribution_index
            )

            merged_v1 = np.concatenate([v1_old, v1, np.zeros_like(o2.x)], axis=0)
            merged_v2 = np.concatenate([v2_old, np.zeros_like(o1.x), v2], axis=0)

            # Line 10-11: 调用 Algorithm 4 环境选择与 Algorithm 5 NGSS 选择
            s1, v1, f1 = algorithm4_environmental_selection(merge_pops(s1, o1, o2), merged_v1, self.n, epsilon)
            s2, v2, f2 = algorithm5_niche_guided_subset_selection(merge_pops(s2, o1, o2), merged_v2, self.n, weights)

            # Line 12-16: 每 T_COL (75) 代触发 Algorithm 3 COL 正交学习
            if t % self.col_frequency == 0:
                tau = problem.eval_count / problem.max_evals
                c1, c2 = algorithm3_collaborative_orthogonal_learning(problem, s1, s2, weights, tau, self.rng)

                merged_v1 = np.concatenate([v1, np.zeros_like(c1.x), np.zeros_like(c2.x)], axis=0)
                merged_v2 = np.concatenate([v2, np.zeros_like(c1.x), np.zeros_like(c2.x)], axis=0)

                s1, v1, f1 = algorithm4_environmental_selection(merge_pops(s1, c1, c2), merged_v1, self.n, epsilon)
                s2, v2, f2 = algorithm5_niche_guided_subset_selection(merge_pops(s2, c1, c2), merged_v2, self.n, weights)

            # Line 17-18: 按照公式 (7) 动态更新 ε
            ratio = float(np.mean(s1.cv <= 1e-12))
            t_max = max(1.0, problem.max_evals / (2.0 * self.n))
            epsilon = update_epsilon(epsilon, ratio, self.alpha, t_max, epsilon_max, self.rng)

            history["fe"].append(float(problem.eval_count))
            history["epsilon"].append(epsilon)
            history["feasible_ratio_s1"].append(ratio)
            history["feasible_ratio_s2"].append(float(np.mean(s2.cv <= 1e-12)))

        # Line 20: 返回 S1 的非支配可行解
        nd = nondominated_feasible_indices(s1.f, s1.cv)
        return Result(
            population=s1,
            feasible_nondominated=take_pop(s1, nd),
            eval_count=problem.eval_count,
            history=history,
        )
