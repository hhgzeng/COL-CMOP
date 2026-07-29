"""CMOCSO (Competitive and Cooperative Swarm Optimizer) Python 实现。

对应 Ming 等人 2023 年 IEEE TEVC 论文及 PlatEMO 源码 CMOCSO.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.cmocso.fitness import cal_fitness
from algorithms.cmocso.operators import competitive_operator, cooperative_operator
from algorithms.cmocso.update_p import update_p, update_p1, update_p2
from core.operators import tournament_selection
from core.schema import CMOP, Array, Population, Result


def update_epsilon(
    tao: float, var_epsilon: float, epsilon_0: float, rf: float, alpha: float, gen: int, tc: float, cp: float
) -> float:
    """更新 epsilon 约束边界值。"""
    if gen > tc:
        return 0.0
    else:
        if rf < alpha:
            return (1.0 - tao) * var_epsilon
        else:
            return epsilon_0 * ((1.0 - (gen / tc)) ** cp)


class CMOCSO:
    """CMOCSO 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        cp: float = 2.0,
        alpha: float = 0.95,
        tao: float = 0.05,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.cp = cp
        self.alpha = alpha
        self.tao = tao
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
        x_init = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        pop = self._evaluate(problem, x_init)

        cv_max = float(np.max(pop.cv)) if len(pop.cv) > 0 else 0.0
        epsilon_0 = cv_max
        var_epsilon = epsilon_0

        comp_pop, comp_fit = update_p1(pop, self.n_pop, var_epsilon)
        coop_pop, coop_fit = update_p2(pop, self.n_pop)
        main_pop = update_p(pop, self.n_pop)

        tc = 0.9 * math.ceil(problem.max_evals / self.n_pop)
        g_max = problem.max_evals / self.n_pop
        y_val = 10.0

        history: dict[str, list[float]] = {"fe": [], "epsilon": []}

        while problem.eval_count < problem.max_evals:
            gen = math.ceil(problem.eval_count / self.n_pop)
            cv_comp = comp_pop.cv
            cv_max_curr = float(np.max(cv_comp)) if len(cv_comp) > 0 else 0.0
            cv_max = max(cv_max_curr, cv_max)
            epsilon_0 = cv_max

            rf = float(np.mean(cv_comp <= 1e-6)) if len(cv_comp) > 0 else 1.0
            var_epsilon = update_epsilon(self.tao, var_epsilon, epsilon_0, rf, self.alpha, gen, tc, self.cp)

            comp_fit = cal_fitness(comp_pop.f, comp_pop.cv, var_epsilon)

            n_comp = len(comp_pop.x)
            if n_comp >= 2:
                perm = self.rng.permutation(n_comp)[: (n_comp // 2) * 2]
            else:
                perm = np.array([0, 0], dtype=int)

            half = len(perm) // 2
            loser_idx = perm[:half]
            winner_idx = perm[half:]

            swap = comp_fit[loser_idx] <= comp_fit[winner_idx]
            loser_final = np.where(swap, winner_idx, loser_idx)
            winner_final = np.where(swap, loser_idx, winner_idx)

            off1 = competitive_operator(problem, comp_pop.x[loser_final], comp_pop.x[winner_final], y_val, self.rng)

            mating = tournament_selection(2, self.n_pop, coop_fit, self.rng)
            off2 = cooperative_operator(problem, coop_pop.x[mating], self.rng)

            merged_off = self._merge(off1, off2)

            main_pop = update_p(self._merge(main_pop, merged_off), self.n_pop)
            comp_pop, comp_fit = update_p1(self._merge(comp_pop, merged_off), self.n_pop, var_epsilon)

            gen = math.ceil(problem.eval_count / self.n_pop)
            y_val = (problem.n_obj ** 2) * ((gen / g_max) - 1.0) ** 2 + 1.0
            coop_pop, coop_fit = update_p2(self._merge(merged_off, coop_pop), self.n_pop)

            history["fe"].append(float(problem.eval_count))
            history["epsilon"].append(float(var_epsilon))

        # Non-dominated feasible solutions
        feas_pop = update_p(main_pop, len(main_pop.x)) if len(main_pop.x) > 0 else main_pop
        final_pop = main_pop if len(main_pop.x) > 0 else comp_pop
        return Result(population=final_pop, feasible_nondominated=feas_pop, eval_count=problem.eval_count, history=history)
