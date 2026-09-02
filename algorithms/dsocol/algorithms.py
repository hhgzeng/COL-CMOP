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
    winner_update_mode: str = "cso",
) -> tuple[Population, Array]:
    """Algorithm 2: Offspring Generation (子代生成)。

    1. 两两随机竞争，分为获胜组 Sw 与失败组 Sl
    2. 调用公式 (4) 更新失败组 Sl 的速度与位置
    3. 群体 1 调用公式 (5) 对 Sw 进行 SBX 交叉；群体 2 调用公式 (6) 最佳引导 Sw (若 winner_update_mode == 'cso')
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

    if winner_update_mode == "cso":
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
    epsilon: float,
    rng: np.random.Generator,
    enable_trend_learning: bool = True,
    enable_orthogonal_learning: bool = True,
) -> tuple[Population | None, Population | None]:
    """Algorithm 3: Collaborative Orthogonal Learning (COL 协同正交学习)。

    1. 基于权重向量与角度选出候选解
    2. 若 enable_trend_learning=True，调用公式 (8) 趋势学习生成主群体的扩展解 main
    3. 若 enable_orthogonal_learning=True，调用公式 (9) 正交补平面对 S2 进行采样生成扩展解 aux
    """
    n = len(s1.x)
    quota = max(1, n // len(weights))
    a1, a2 = calc_angles(s1.f, weights), calc_angles(s2.f, weights)
    main, aux = [], []
    lower, upper = problem.lower, problem.upper

    for j in range(len(weights)):
        # S1 选出候选解，划分为 epsilon 可行非支配集与剩余解集 (Paper Alg 3 Line 5)
        cand1 = np.argsort(a1[:, j])[:quota]
        c1_cv = s1.cv[cand1]
        relaxed_mask = c1_cv <= epsilon + 1e-12
        relaxed_idx = cand1[relaxed_mask]
        rem_idx = cand1[~relaxed_mask]

        if len(relaxed_idx) > 0 and len(rem_idx) > 0:
            winner = rng.choice(relaxed_idx)
            loser = rng.choice(rem_idx)
        else:
            ranked = cand1[np.argsort(s1.cv[cand1])]
            mid = max(1, len(ranked) // 2)
            winner = rng.choice(ranked[:mid])
            loser = rng.choice(ranked[mid:]) if mid < len(ranked) else winner

        boundary = upper if rng.random() < 0.5 else lower

        if enable_trend_learning:
            # S1 趋势学习调用公式 (8): u_main = x_r + eta_k * v
            direction = compute_col_trend_direction(s1.x[winner], s1.x[loser], boundary, tau)
            dist = np.linalg.norm(s1.x[winner] - boundary)
            main.append(boundary + rng.uniform(0.5, 1.5) * dist * direction)
        else:
            # 无趋势学习时生成随机单位方向作为正交补基准方向
            v_rand = rng.normal(size=len(lower))
            norm = np.linalg.norm(v_rand)
            direction = v_rand / norm if norm > 1e-15 else v_rand

        if enable_orthogonal_learning:
            # S2 正交补学习调用公式 (9)
            cand2 = np.argsort(a2[:, j])[:quota]
            feasible_nd = cand2[nondominated_feasible_indices(s2.f[cand2], s2.cv[cand2])]
            anchor = (
                rng.choice(feasible_nd)
                if len(feasible_nd)
                else cand2[np.argmin(s2.cv[cand2])]
            )
            perpendicular = compute_col_orthogonal_direction(direction, rng)
            aux.append(s2.x[anchor] + rng.uniform(-1, 1) * perpendicular)

    pop_main = None
    if main:
        main_arr = np.clip(np.asarray(main), lower, upper)
        main_arr = polynomial_mutation(main_arr, lower, upper, rng)
        res_main = problem.evaluate(main_arr)
        pop_main = Population(x=main_arr, f=res_main.f, cv=res_main.cv, g=res_main.g, h=res_main.h)

    pop_aux = None
    if aux:
        aux_arr = np.clip(np.asarray(aux), lower, upper)
        aux_arr = polynomial_mutation(aux_arr, lower, upper, rng)
        res_aux = problem.evaluate(aux_arr)
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
    """Algorithm 5: Niche-Guided Subset Selection (NGSS 生态位引导子集选择)。

    严格对应论文 Algorithm 5 三层子集划分策略：
    - First-Level (Lines 1-3): 计算各解与权重向量夹角，按 CDP 划分为非支配可行解集 Phi_nf 与剩余集 Phi_rem
    - Second-Level (Lines 4-9): 遍历生态位，若超出容量 L = floor(N/K)，保留拥挤度最好的 L 个解，多余解加入 Phi_rem
    - Third-Level (Lines 10-20): 遍历生态位，若低于容量 L：
        * 若该生态位自身的剩余解池 Phi_j_rem > 需求量，按 CDP 挑选补充；
        * 否则全部加入 Phi_j_rem，并从全局剩余池 Phi_rem 中挑选到第 j 个生态位向量夹角最小的解借调补充至 L。
    - Line 21-23: 合并各生态位解集，评估 SPEA2 适应度 (epsilon=0，即 CDP)。
    """
    k = len(weights)
    capacity = max(1, target_n // k)
    angle = calc_angles(pop.f, weights)
    niche = angle.argmin(axis=1)

    # Line 2: 按 CDP 划分为非支配可行集 Phi_nf 与剩余候选集 Phi_rem
    nd = set(nondominated_feasible_indices(pop.f, pop.cv).tolist())
    remainder = set(i for i in range(len(pop.x)) if i not in nd)

    # 计算 CDP 适应度 (ε=0 的 SPEA2 适应度，用于在 CDP 准则下排序)
    cdp_fitness = spea2_fitness(pop.f, pop.cv, 0.0)

    # Line 3-9: Second-Level Subset Division
    niche_nf: list[list[int]] = [[] for _ in range(k)]
    for i in range(k):
        members = [idx for idx in nd if niche[idx] == i]
        if len(members) > capacity:
            # 内部拥挤度距离：选择最小距离最大的 capacity 个个体 (最稀疏/最优拥挤度)
            d = np.linalg.norm(
                pop.f[members, None] - pop.f[np.array(members)][None], axis=2
            )
            np.fill_diagonal(d, np.inf)
            keep = np.asarray(members)[
                np.argsort(d.min(axis=1))[::-1][:capacity]
            ].tolist()
            unselected = [idx for idx in members if idx not in keep]
            remainder.update(unselected)
            members = keep
        niche_nf[i] = members

    # Line 10-20: Third-Level Subset Division
    for j in range(k):
        need = capacity - len(niche_nf[j])
        if need <= 0:
            continue

        # 当前生态位自有的剩余候选集 Phi_j^rem
        phi_j_rem = [idx for idx in remainder if niche[idx] == j]

        if len(phi_j_rem) > need:
            # Line 14: 根据 CDP 选择 need 个解补充到 Phi_j^nf
            phi_j_rem.sort(key=lambda idx: (pop.cv[idx] > 1e-12, cdp_fitness[idx], angle[idx, j]))
            chosen = phi_j_rem[:need]
            niche_nf[j].extend(chosen)
            remainder.difference_update(chosen)
        else:
            # Line 16: 全部加入 Phi_j^rem
            niche_nf[j].extend(phi_j_rem)
            remainder.difference_update(phi_j_rem)
            shortage = capacity - len(niche_nf[j])

            # Line 17: 从全局剩余池 Phi^rem 中选择到第 j 个生态位夹角最小的解借调补充
            if shortage > 0 and len(remainder) > 0:
                rem_candidates = sorted(remainder, key=lambda idx: (angle[idx, j], cdp_fitness[idx]))
                borrowed = rem_candidates[:shortage]
                niche_nf[j].extend(borrowed)
                remainder.difference_update(borrowed)

    # Line 21: 合并所有生态位的解
    selected = [idx for sublist in niche_nf for idx in sublist]

    # 当 target_n 不能被 k 整除时的尾数微调补充 (例如 target_n=100, k=11, 11*9=99 差 1 个)
    if len(selected) < target_n and len(remainder) > 0:
        extra_candidates = sorted(remainder, key=lambda idx: (pop.cv[idx] > 1e-12, cdp_fitness[idx]))
        selected.extend(extra_candidates[: target_n - len(selected)])

    idx = np.asarray(selected[:target_n], dtype=int)
    out_pop = take_pop(pop, idx)
    selected_vel = (
        vel[idx] if vel is not None and len(vel) == len(pop.x) else np.zeros_like(out_pop.x)
    )
    return out_pop, selected_vel, spea2_fitness(out_pop.f, out_pop.cv, 0.0)


# ===========================================================================
# Algorithm 1: The Framework of DSOCOL (DSOCOL 主框架及消融变体)
# ===========================================================================

class DSOCOL:
    """Algorithm 1: The framework of DSOCOL (支持控制开关及消融扩展)。"""

    def __init__(
        self,
        population_size: int = 100,
        col_frequency: int = 75,
        alpha: float = 0.95,
        distribution_index: float = 20.0,
        seed: int | None = None,
        use_ngss: bool = True,
        winner_update_mode: str = "cso",
        enable_col: bool = True,
        enable_trend_learning: bool = True,
        enable_orthogonal_learning: bool = True,
    ):
        if population_size < 4 or population_size % 2:
            raise ValueError("种群规模必须是大于等于 4 的偶数")
        self.n = population_size
        self.col_frequency = col_frequency
        self.alpha = alpha
        self.distribution_index = distribution_index
        self.rng = np.random.default_rng(seed)

        # 消融控制开关
        self.use_ngss = use_ngss
        self.winner_update_mode = winner_update_mode
        self.enable_col = enable_col
        self.enable_trend_learning = enable_trend_learning
        self.enable_orthogonal_learning = enable_orthogonal_learning

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
        ref_front = getattr(problem, "ref_front", None)
        if ref_front is not None:
            history["igd"] = []

        t = 0
        total_generations = max(1, problem.max_evals // (2 * self.n))
        tc = max(1, total_generations // 3)  # noqa

        # Line 7: while t <= T (在此通过 problem.eval_count < problem.max_evals 精确驱动)
        while problem.eval_count < problem.max_evals:
            t += 1

            v1_old, v2_old = v1.copy(), v2.copy()

            # Line 8-9: 调用 Algorithm 2 生成子代 O1, O2
            o1, v1 = algorithm2_offspring_generation(
                problem, s1, v1, f1, 1, self.rng, self.distribution_index, self.winner_update_mode
            )
            o2, v2 = algorithm2_offspring_generation(
                problem, s2, v2, f2, 2, self.rng, self.distribution_index, self.winner_update_mode
            )

            merged_v1 = np.concatenate([v1_old, v1, np.zeros_like(o2.x)], axis=0)
            merged_v2 = np.concatenate([v2_old, np.zeros_like(o1.x), v2], axis=0)

            # Line 10-11: 环境选择 (S1 使用 SPEA2, S2 根据 use_ngss 选用 NGSS 或 SPEA2)
            s1, v1, f1 = algorithm4_environmental_selection(merge_pops(s1, o1, o2), merged_v1, self.n, epsilon)
            if self.use_ngss:
                s2, v2, f2 = algorithm5_niche_guided_subset_selection(merge_pops(s2, o1, o2), merged_v2, self.n, weights)
            else:
                s2, v2, f2 = algorithm4_environmental_selection(merge_pops(s2, o1, o2), merged_v2, self.n, 0.0)

            # Line 12-16: 每 T_COL (75) 代触发 Algorithm 3 COL 协同正交学习
            if self.enable_col and t % self.col_frequency == 0:
                tau = problem.eval_count / problem.max_evals
                c1, c2 = algorithm3_collaborative_orthogonal_learning(
                    problem,
                    s1,
                    s2,
                    weights,
                    tau,
                    epsilon,
                    self.rng,
                    enable_trend_learning=self.enable_trend_learning,
                    enable_orthogonal_learning=self.enable_orthogonal_learning,
                )

                col_pops = [p for p in (c1, c2) if p is not None]
                if col_pops:
                    merged_c = merge_pops(*col_pops)
                    zeros_c = np.zeros_like(merged_c.x)
                    merged_v1 = np.concatenate([v1, zeros_c], axis=0)
                    merged_v2 = np.concatenate([v2, zeros_c], axis=0)

                    s1, v1, f1 = algorithm4_environmental_selection(merge_pops(s1, merged_c), merged_v1, self.n, epsilon)
                    if self.use_ngss:
                        s2, v2, f2 = algorithm5_niche_guided_subset_selection(merge_pops(s2, merged_c), merged_v2, self.n, weights)
                    else:
                        s2, v2, f2 = algorithm4_environmental_selection(merge_pops(s2, merged_c), merged_v2, self.n, 0.0)

            # Line 17-18: 按照公式 (7) 动态更新 ε
            ratio = float(np.mean(s1.cv <= 1e-12))
            t_max = max(1.0, problem.max_evals / (2.0 * self.n))
            tc = max(1.0, t_max / 3.0)
            epsilon = update_epsilon(t, tc, epsilon, ratio, self.alpha, epsilon_max, self.rng)

            history["fe"].append(float(problem.eval_count))
            history["epsilon"].append(epsilon)
            history["feasible_ratio_s1"].append(ratio)
            history["feasible_ratio_s2"].append(float(np.mean(s2.cv <= 1e-12)))

            if ref_front is not None:
                nd_temp = nondominated_feasible_indices(s1.f, s1.cv)
                if len(nd_temp) > 0:
                    from core.metrics import calculate_igd
                    history["igd"].append(calculate_igd(s1.f[nd_temp], ref_front))
                else:
                    history["igd"].append(float("nan"))

        # Line 20: 返回 S1 的非支配可行解
        nd = nondominated_feasible_indices(s1.f, s1.cv)
        return Result(
            population=s1,
            feasible_nondominated=take_pop(s1, nd),
            eval_count=problem.eval_count,
            history=history,
        )


class DSOCOL1(DSOCOL):
    """消融变体 DSOCOL1：辅助种群 S2 采用 Algorithm 4 (SPEA2 截断) 替代 NGSS 策略。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(use_ngss=False, **kwargs)


class DSOCOL2(DSOCOL):
    """消融变体 DSOCOL2：两种群获胜组仅通过多项式变异更新。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(winner_update_mode="mutation", **kwargs)


class DSOCOL3(DSOCOL):
    """消融变体 DSOCOL3：不实现 COL (协同正交学习) 策略。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_col=False, **kwargs)


class DSOCOL4(DSOCOL):
    """消融变体 DSOCOL4：不实现趋势学习策略 (移除 Eq. (8))。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_trend_learning=False, **kwargs)


class DSOCOL5(DSOCOL):
    """消融变体 DSOCOL5：不实现正交学习策略 (移除 Eq. (9))。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_orthogonal_learning=False, **kwargs)

