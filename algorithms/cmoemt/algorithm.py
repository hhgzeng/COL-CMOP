"""CMOEMT (Constrained Multi-objective Optimization via Multitasking and Knowledge Transfer) Python 实现。

对应 Ming 等人 2024 年 IEEE TEVC 论文及 PlatEMO 源码 CMOEMT.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.cmoemt.fitness import cal_fitness
from algorithms.cmoemt.selection import environmental_selection_t1, environmental_selection_t3
from core.operators import operator_de, operator_ga, tournament_selection, uniform_point
from core.schema import CMOP, Array, Population, Result


def calc_maxchange(ideal_points: Array, nadir_points: Array, gen: int, last_gen: int) -> float:
    """计算理想点/Nadir点离散最大变化率。"""
    delta_value = 1e-6 * np.ones(ideal_points.shape[1], dtype=float)
    prev_idx = gen - last_gen
    denom_ideal = np.maximum(ideal_points[prev_idx], delta_value)
    denom_nadir = np.maximum(nadir_points[prev_idx], delta_value)
    rz = np.abs((ideal_points[gen] - ideal_points[prev_idx]) / denom_ideal)
    nrz = np.abs((nadir_points[gen] - nadir_points[prev_idx]) / denom_nadir)
    return float(np.max([rz.max(), nrz.max()]))


def update_epsilon(
    tao: float, epsilon_k: float, epsilon_0: float, rf: float, alpha1: float, gen: int, tc: float, cp: float
) -> float:
    """动态更新 epsilon 约束边界值。"""
    if rf < alpha1:
        return (1.0 - tao) * epsilon_k
    else:
        return epsilon_0 * ((1.0 - (gen / tc)) ** cp)


class CMOEMT:
    """CMOEMT 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        delta: float = 0.9,
        nr: int = 2,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.delta = delta
        self.nr = nr
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, problem: CMOP, x: Array) -> Population:
        res = problem.evaluate(x)
        return Population(x=np.asarray(x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)

    @staticmethod
    def _merge(*pops: Population) -> Population:
        x_concat = np.concatenate([p.x for p in pops if len(p.x) > 0], axis=0)
        f_concat = np.concatenate([p.f for p in pops if len(p.f) > 0], axis=0)
        cv_concat = np.concatenate([p.cv for p in pops if len(p.cv) > 0], axis=0)
        valid_g = [p.g for p in pops if p.g is not None and len(p.g) > 0]
        g_concat = np.concatenate(valid_g, axis=0) if valid_g else None
        valid_h = [p.h for p in pops if p.h is not None and len(p.h) > 0]
        h_concat = np.concatenate(valid_h, axis=0) if valid_h else None
        return Population(x=x_concat, f=f_concat, cv=cv_concat, g=g_concat, h=h_concat)

    def run(self, problem: CMOP) -> Result:
        lo, hi = problem.lower, problem.upper
        m = problem.n_obj

        # 1. 权重向量与邻域结构
        weights, real_n = uniform_point(self.n_pop, m)
        self.n_pop = real_n
        t = math.ceil(self.n_pop / 10.0)

        dist_w = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        b_neighbors = np.argsort(dist_w, axis=1)[:, :t]

        # 2. 初始化三大任务群体
        x1 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        x2 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        x3 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))

        pop1 = self._evaluate(problem, x1)
        pop2 = self._evaluate(problem, x2)
        pop3 = self._evaluate(problem, x3)

        z_ideal = np.min(pop2.f, axis=0)
        fit1 = cal_fitness(pop1.f, pop1.g)
        fit3 = cal_fitness(pop3.f, None)

        tc = 0.9 * math.ceil(problem.max_evals / self.n_pop)
        last_gen = 20
        change_threshold = 0.1
        search_stage = 1
        max_change = 1.0
        epsilon_k = 0.0
        epsilon_0 = 0.0
        cp = 2.0
        alpha1 = 0.95
        tao = 0.05

        max_g = math.ceil(problem.max_evals / (2 * self.n_pop)) + 100
        ideal_points = np.zeros((max_g, m), dtype=float)
        nadir_points = np.zeros((max_g, m), dtype=float)

        history: dict[str, list[float]] = {"fe": [], "search_stage": []}

        while problem.eval_count < problem.max_evals:
            gen = math.ceil(problem.eval_count / (2 * self.n_pop)) - 1
            gen = max(0, gen)

            cv2 = pop2.cv
            rf = float(np.mean(cv2 <= 1e-6))

            ideal_points[gen] = z_ideal
            nadir_points[gen] = np.max(pop2.f, axis=0)

            if gen >= last_gen:
                max_change = calc_maxchange(ideal_points, nadir_points, gen, last_gen)

            if gen < tc:
                if max_change <= change_threshold and search_stage == 1:
                    search_stage = -1
                    epsilon_0 = float(np.max(cv2)) if len(cv2) > 0 else 0.0
                    epsilon_k = epsilon_0
                if search_stage == -1:
                    epsilon_k = update_epsilon(tao, epsilon_k, epsilon_0, rf, alpha1, gen, tc, cp)
            else:
                epsilon_k = 0.0

            length_o1 = self.n_pop // 2
            length_o3 = self.n_pop // 2

            mating1 = tournament_selection(2, 2 * length_o1, fit1, self.rng)
            off1_x = operator_ga(pop1.x[mating1], lo, hi, self.rng)
            off1 = self._evaluate(problem, off1_x)

            # Task 2 (MOEA/D) 局部/全局 DE 演化
            off2_list: list[Population] = []
            for _ in range(5):
                boundary = np.flatnonzero(np.sum(weights < 1e-3, axis=1) == m - 1)
                boundary_list = list(boundary) + [len(weights) // 2]
                rand_indices = list(self.rng.choice(len(weights), size=max(1, len(weights) // 5 - len(boundary_list)), replace=True))
                sub_indices = boundary_list + rand_indices

                for i in sub_indices:
                    i = i % self.n_pop
                    if self.rng.random() < self.delta:
                        p_pool = b_neighbors[i, self.rng.permutation(t)]
                    else:
                        p_pool = self.rng.permutation(self.n_pop)

                    p1_idx = p_pool[0]
                    p2_idx = p_pool[1]

                    off_x_sub = operator_de(
                        pop2.x[i : i + 1],
                        pop2.x[p1_idx : p1_idx + 1],
                        pop2.x[p2_idx : p2_idx + 1],
                        lo,
                        hi,
                        self.rng,
                    )
                    off_sub = self._evaluate(problem, off_x_sub)
                    off2_list.append(off_sub)

                    z_ideal = np.minimum(z_ideal, off_sub.f[0])

                    # TCH 标量化比较与替换
                    g_old = np.max(np.abs(pop2.f[p_pool] - z_ideal) * weights[p_pool], axis=1)
                    g_new = np.max(np.abs(off_sub.f[0] - z_ideal) * weights[p_pool], axis=1)
                    cv_old = pop2.cv[p_pool]
                    cv_new = off_sub.cv[0] * np.ones(len(p_pool))

                    if search_stage == 1:  # Push stage
                        cond = g_old >= g_new
                    else:  # Pull stage with epsilon
                        cond = ((g_old >= g_new) & (((cv_old <= epsilon_k) & (cv_new <= epsilon_k)) | (cv_old == cv_new))) | (cv_new < cv_old)

                    replace_indices = p_pool[np.flatnonzero(cond)[: self.nr]]
                    for r_idx in replace_indices:
                        pop2.x[r_idx] = off_sub.x[0]
                        pop2.f[r_idx] = off_sub.f[0]
                        pop2.cv[r_idx] = off_sub.cv[0]
                        if pop2.g is not None and off_sub.g is not None:
                            pop2.g[r_idx] = off_sub.g[0]

            off2 = self._merge(*off2_list) if len(off2_list) > 0 else Population(x=np.empty((0, len(lo))), f=np.empty((0, m)), cv=np.empty(0))

            mating3 = tournament_selection(2, 2 * length_o3, fit3, self.rng)
            off3_x = operator_ga(pop3.x[mating3], lo, hi, self.rng)
            off3 = self._evaluate(problem, off3_x)

            # 知识共享阶段判断
            if problem.eval_count < problem.max_evals / 2:
                pop1, fit1 = environmental_selection_t1(self._merge(pop1, off1, off2, off3), self.n_pop)
                pop3, fit3 = environmental_selection_t3(self._merge(pop3, off1, off2, off3), self.n_pop)
            else:
                pop1, fit1 = environmental_selection_t1(self._merge(pop1, pop2, pop3, off1, off2, off3), self.n_pop)
                pop3, fit3 = environmental_selection_t3(self._merge(pop1, pop2, pop3, off1, off2, off3), self.n_pop)

            history["fe"].append(float(problem.eval_count))
            history["search_stage"].append(float(search_stage))

        # Final non-dominated feasible population
        feas_mask = pop1.cv <= 1e-12
        if np.any(feas_mask):
            f_feas = pop1.f[feas_mask]
            n_f = len(f_feas)
            dom = np.zeros((n_f, n_f), dtype=bool)
            for i in range(n_f):
                for j in range(n_f):
                    if i != j and np.all(f_feas[i] <= f_feas[j]) and np.any(f_feas[i] < f_feas[j]):
                        dom[i, j] = True
            nd_idx = np.flatnonzero(np.sum(dom, axis=0) == 0)
            final_idx = np.flatnonzero(feas_mask)[nd_idx]
            nondom_pop = Population(x=pop1.x[final_idx], f=pop1.f[final_idx], cv=pop1.cv[final_idx])
        else:
            nondom_pop = Population(x=np.empty((0, len(lo))), f=np.empty((0, m)), cv=np.empty(0))

        return Result(population=pop1, feasible_nondominated=nondom_pop, eval_count=problem.eval_count, history=history)
