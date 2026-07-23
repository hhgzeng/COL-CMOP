"""DVCEA (Decision Variables Classification-based Evolutionary Algorithm) Python 实现。

对应 Ban 等人 2025 年 IEEE/CAA JAS 论文及 PlatEMO 源码 DVCEA.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.dvcea.classification import kmeans_clusters, variable_classification
from algorithms.dvcea.fitness import cal_fitness_e
from algorithms.dvcea.operators import degenerator_better, operator_de_pbest_1_main
from algorithms.dvcea.selection import improve_e_environmental_selection
from core.schema import CMOP, Array, Population, Result


class DVCEA:
    """DVCEA 算法主类。"""

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
        x_init = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        pop = self._evaluate(problem, x_init)

        centers = kmeans_clusters(pop.x, k=5, rng=self.rng)
        fea_idx, infea_idx = variable_classification(problem, pop, centers, self.rng)

        cv_max = float(np.max(pop.cv)) if len(pop.cv) > 0 else 0.0
        epsilon_0 = 1.0 if cv_max == 0 else cv_max
        fit = cal_fitness_e(pop.f, pop.cv, epsilon_0)

        history: dict[str, list[float]] = {"fe": [], "epsilon": []}

        while problem.eval_count < problem.max_evals:
            fe_ratio = problem.eval_count / problem.max_evals
            cp = (-math.log(epsilon_0) - 6.0) / math.log(1.0 - 0.5 + 1e-15)
            var_epsilon = epsilon_0 * ((1.0 - fe_ratio) ** cp)

            off1 = operator_de_pbest_1_main(pop, problem, fit, fea_idx, 0.1, self.rng)
            pop, _ = improve_e_environmental_selection(self._merge(pop, off1), self.n_pop, var_epsilon)

            off2 = degenerator_better(pop, problem, infea_idx, var_epsilon, self.rng)
            pop, fit = improve_e_environmental_selection(self._merge(pop, off2), self.n_pop, var_epsilon)

            history["fe"].append(float(problem.eval_count))
            history["epsilon"].append(float(var_epsilon))

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
