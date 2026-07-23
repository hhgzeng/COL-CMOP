"""LCMEA (Large-scale Constrained Multi-objective Evolutionary Algorithm) Python 实现。

对应 Si 等人 2025 年 IEEE TETCI 论文及 PlatEMO 源码 LCMEA.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.lcmea.esp import esp_offspring_generation
from algorithms.lcmea.selection import env_cdp_selection, env_epsilon_selection
from core.schema import CMOP, Array, Population, Result


class LCMEA:
    """LCMEA 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        seed: int | None = None,
    ):
        self.n_pop = population_size
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

        # 1. 初始化归档集 Archive (2 * N)
        arch_x = self.rng.uniform(lo, hi, size=(2 * self.n_pop, len(lo)))
        archive = self._evaluate(problem, arch_x)

        cv_max = float(np.max(archive.cv)) if len(archive.cv) > 0 else 0.0
        var_0 = 1.0 if cv_max == 0 else cv_max

        x_val = 0.0
        cp = (-math.log(var_0) - 6.0) / math.log(1.0 - 0.5 + 1e-15)
        var_epsilon = var_0 * ((1.0 - x_val) ** cp)

        pop = env_cdp_selection(archive, self.n_pop)
        pre_action = -1
        action = 0

        history: dict[str, list[float]] = {"fe": [], "action": []}

        while problem.eval_count < problem.max_evals:
            offspring = esp_offspring_generation(pop, problem, self.rng)
            merged_pop = self._merge(pop, offspring)

            # RL 动态策略动作选择 (0: CDP, 1: Epsilon)
            action = int(self.rng.integers(0, 2)) if problem.eval_count < problem.max_evals * 0.5 else 0

            if action == 0:
                pop = env_cdp_selection(merged_pop, self.n_pop)
            else:
                pop = env_epsilon_selection(merged_pop, self.n_pop, var_epsilon)

            if action != pre_action:
                pop = env_cdp_selection(self._merge(archive, pop), self.n_pop)
            pre_action = action

            var_epsilon = var_0 * ((1.0 - x_val) ** cp)
            if var_epsilon < 1e-6:
                var_epsilon = 0.0
            x_val += 1.0 / max(1.0, (problem.max_evals / self.n_pop))

            history["fe"].append(float(problem.eval_count))
            history["action"].append(float(action))

        # Final non-dominated feasible population
        feas_mask = pop.cv <= 1e-12
        if np.any(feas_mask):
            f_feas = pop.f[feas_mask]
            n_f = len(f_feas)
            dom = np.zeros((n_f, n_f), dtype=bool)
            for i in range(n_f):
                for j in range(n_f):
                    if i != j and np.all(f_feas[i] <= f_feas[j]) and np.any(f_feas[i] < f_feas[j]):
                        dom[i, j] = True
            nd_idx = np.flatnonzero(np.sum(dom, axis=0) == 0)
            final_idx = np.flatnonzero(feas_mask)[nd_idx]
            nondom_pop = Population(x=pop.x[final_idx], f=pop.f[final_idx], cv=pop.cv[final_idx])
        else:
            nondom_pop = Population(x=np.empty((0, len(lo))), f=np.empty((0, problem.n_obj)), cv=np.empty(0))

        return Result(population=pop, feasible_nondominated=nondom_pop, eval_count=problem.eval_count, history=history)
