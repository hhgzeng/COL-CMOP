"""APSEA (Adaptive Population Sizing based Evolutionary Algorithm) 算法。

对应 Tian 等人 2025 年 Neurocomputing 论文及 PlatEMO 源码 APSEA.m。
接入统一 CMOP 接口 (PymooProblemAdapter)，基于精确 FE 预算控制演化。
"""
from __future__ import annotations

import math
import numpy as np

from core.operators import operator_ga, tournament_selection
from core.schema import CMOP, Array, Population, Result
from algorithms.apsea.fitness import cal_fitness
from algorithms.apsea.selection import (
    environmental_selection_cdp,
    environmental_selection_epsilon,
    environmental_selection_no_constrained,
)


def calc_maxchange(ideal_points: Array, nadir_points: Array, gen: int, last_gen: int) -> float:
    """计算目标理想点与 Nadir 点的离散最大变化率 (对应 APSEA.m)。"""
    delta_value = 1e-6 * np.ones(ideal_points.shape[1], dtype=float)
    prev_idx = gen - last_gen  # 下标从 0 开始
    
    denom_ideal = np.maximum(ideal_points[prev_idx], delta_value)
    denom_nadir = np.maximum(nadir_points[prev_idx], delta_value)

    rz = np.abs((ideal_points[gen] - ideal_points[prev_idx]) / denom_ideal)
    nrz = np.abs((nadir_points[gen] - nadir_points[prev_idx]) / denom_nadir)
    return float(np.max([rz.max(), nrz.max()]))


def reduce_boundary(ef: float, k: int, max_k: int, cp: float) -> float:
    """计算动态 epsilon 约束边界 (对应 APSEA.m 的 ReduceBoundary)。"""
    z = 1e-8
    nearzero = 1e-15
    log_val = np.log((ef + z) / z)
    b = max_k / (np.power(log_val, 1.0 / cp) + nearzero)
    f = ef * np.exp(-np.power(k / b, cp))
    epsn = f - z
    return float(max(0.0, epsn))


class APSEA:
    """APSEA 算法框架实现。"""

    def __init__(
        self,
        population_size: int = 100,
        alpha: float = 0.05,
        beta: float = 0.05,
        cp: float = 5.0,
        last_gen: int = 20,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.alpha = alpha
        self.beta = beta
        self.cp = cp
        self.last_gen = last_gen
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, problem: CMOP, x: Array) -> Population:
        res = problem.evaluate(x)
        return Population(x=np.asarray(x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)

    @staticmethod
    def _merge(*pops: Population) -> Population:
        x_concat = np.concatenate([p.x for p in pops], axis=0)
        f_concat = np.concatenate([p.f for p in pops], axis=0)
        cv_concat = np.concatenate([p.cv for p in pops], axis=0)
        g_concat = np.concatenate([p.g for p in pops if p.g is not None], axis=0) if pops[0].g is not None else None
        h_concat = np.concatenate([p.h for p in pops if p.h is not None], axis=0) if pops[0].h is not None else None
        return Population(x=x_concat, f=f_concat, cv=cv_concat, g=g_concat, h=h_concat)

    @staticmethod
    def _take(pop: Population, idx: Array) -> Population:
        return Population(
            x=pop.x[idx],
            f=pop.f[idx],
            cv=pop.cv[idx],
            g=pop.g[idx] if pop.g is not None else None,
            h=pop.h[idx] if pop.h is not None else None,
        )

    def run(self, problem: CMOP) -> Result:
        """运行 APSEA 算法。"""
        lo, hi = problem.lower, problem.upper
        
        # 1. 初始化双群体 Population1 & Population2
        x1 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        x2 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))

        pop1 = self._evaluate(problem, x1)
        pop2 = self._evaluate(problem, x2)

        fit1 = cal_fitness(pop1.f, pop1.g)
        fit2 = cal_fitness(pop2.f, None)

        max_gen_estimate = max(1, problem.max_evals // self.n_pop)
        ideal_points = np.zeros((max_gen_estimate + 100, problem.n_obj), dtype=float)
        nadir_points = np.zeros((max_gen_estimate + 100, problem.n_obj), dtype=float)

        epsilon0 = float(np.max(pop2.cv))
        if epsilon0 == 0:
            epsilon0 = 1.0

        gen = 0
        history: dict[str, list[float]] = {
            "fe": [],
            "fr": [],
            "sub_pop2_size": [],
        }

        while problem.eval_count < problem.max_evals:
            fr = float(np.mean(pop1.cv <= 1e-12))
            
            # 记录理想点与 Nadir 点
            ideal_points[gen] = np.min(pop1.f, axis=0)
            nadir_points[gen] = np.max(pop1.f, axis=0)

            max_change = 0.0
            if gen >= self.last_gen:
                max_change = calc_maxchange(ideal_points, nadir_points, gen, self.last_gen)

            fe_ratio = problem.eval_count / problem.max_evals

            # 三大自适应分支逻辑
            if fr <= self.alpha or gen < self.last_gen:
                # 分支 1：低可行率或前期阶段
                n_sub2 = max(
                    int(math.ceil(self.n_pop / 2.0 * (1.0 - math.log2(1.0 + fr)) + self.n_pop / 2.0 * (1.0 - math.log2(1.0 + fe_ratio)))),
                    2,
                )

                mating1 = tournament_selection(2, self.n_pop, fit1, self.rng)
                mating2 = tournament_selection(2, n_sub2, fit2, self.rng)

                off1_x = operator_ga(pop1.x[mating1], lo, hi, self.rng)
                off2_x = operator_ga(pop2.x[mating2], lo, hi, self.rng)

                off1 = self._evaluate(problem, off1_x)
                off2 = self._evaluate(problem, off2_x)

                pop1, fit1 = environmental_selection_cdp(self._merge(pop1, off1, off2), self.n_pop)
                pop2, fit2 = environmental_selection_no_constrained(self._merge(pop2, off1, off2), n_sub2)

            elif max_change > self.beta:
                # 分支 2：目标空间剧烈变化阶段 -> epsilon 约束控制
                max_k = max(1, math.ceil(problem.max_evals / self.n_pop) - 1)
                epsilon = reduce_boundary(epsilon0, gen, max_k, self.cp)

                n_sub2 = max(
                    int(math.ceil(self.n_pop / 2.0 * (1.0 - math.log2(1.0 + fr)) + self.n_pop / 2.0 * (1.0 - math.log2(1.0 + fe_ratio)))),
                    2,
                )

                mating1 = tournament_selection(2, self.n_pop, fit1, self.rng)
                mating2 = tournament_selection(2, n_sub2, fit2, self.rng)

                off1_x = operator_ga(pop1.x[mating1], lo, hi, self.rng)
                off2_x = operator_ga(pop2.x[mating2], lo, hi, self.rng)

                off1 = self._evaluate(problem, off1_x)
                off2 = self._evaluate(problem, off2_x)

                pop1, fit1 = environmental_selection_cdp(self._merge(pop1, off1, off2), self.n_pop)
                pop2, fit2 = environmental_selection_epsilon(self._merge(pop2, off1, off2), n_sub2, epsilon)

            else:
                # 分支 3：收敛稳定期 -> 集中资源更新主群体 Pop1
                n_sub2 = len(pop2.x)
                mating1 = tournament_selection(2, self.n_pop, fit1, self.rng)
                off1_x = operator_ga(pop1.x[mating1], lo, hi, self.rng)
                off1 = self._evaluate(problem, off1_x)

                pop1, fit1 = environmental_selection_cdp(self._merge(pop1, off1), self.n_pop)

            history["fe"].append(float(problem.eval_count))
            history["fr"].append(fr)
            history["sub_pop2_size"].append(float(n_sub2))

            gen += 1

        # 计算主群体 Pop1 的非支配可行解
        feasible_mask = pop1.cv <= 1e-12
        if np.any(feasible_mask):
            f_feasible = pop1.f[feasible_mask]
            # 计算可行解之间的非支配关系
            n_f = len(f_feasible)
            dom = np.zeros((n_f, n_f), dtype=bool)
            for i in range(n_f):
                for j in range(n_f):
                    if i != j:
                        dom[i, j] = np.all(f_feasible[i] <= f_feasible[j]) and np.any(f_feasible[i] < f_feasible[j])
            nd_indices = np.flatnonzero(dom.sum(axis=0) == 0)
            feasible_idx_global = np.flatnonzero(feasible_mask)[nd_indices]
            nondom_pop = self._take(pop1, feasible_idx_global)
        else:
            nondom_pop = Population(
                x=np.empty((0, len(lo))),
                f=np.empty((0, problem.n_obj)),
                cv=np.empty(0),
            )

        return Result(
            population=pop1,
            feasible_nondominated=nondom_pop,
            eval_count=problem.eval_count,
            history=history,
        )
