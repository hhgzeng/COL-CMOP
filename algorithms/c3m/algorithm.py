"""C3M (Constraint, Multi-objective, Multi-stage, Multi-constraint Evolutionary Algorithm) Python 实现。

对应 Sun 等人 2023 年 IEEE TEVC 论文及 PlatEMO 源码 C3M.m。
"""
from __future__ import annotations

import numpy as np

from algorithms.c3m.archive import archive_selection
from algorithms.c3m.fitness import cal_fitness
from algorithms.c3m.selection import environmental_selection
from core.operators import operator_de, operator_ga, tournament_selection
from core.schema import CMOP, Array, Population, Result


class C3M:
    """C3M 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        operator_type: int = 1,
        change_threshold: float = 1e-3,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.operator_type = operator_type
        self.change_threshold = change_threshold
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

        totalcon = pop.g.shape[1] if pop.g is not None else 0
        processcon = 0
        fit = cal_fitness(pop.f, pop.g, processcon, totalcon)
        arch = pop

        pops_sub: list[dict] = []
        for i in range(1, totalcon + 1):
            pops_sub.append({
                "pop": pop,
                "fit": cal_fitness(pop.f, pop.g, i, totalcon),
                "idx": i,
                "processed": 0,
            })

        obj_values = [float(np.sum(np.abs(pop.f)))]
        flag = True
        ns = False
        seq: list[int] = []
        index = 0
        processed: list[int] = []

        history: dict[str, list[float]] = {"fe": [], "processcon": []}

        while problem.eval_count < problem.max_evals:
            if processcon <= totalcon and ns:
                ns = False
                x_reinit = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
                pop = self._evaluate(problem, x_reinit)
                fit = cal_fitness(pop.f, pop.g, processcon, totalcon)

            if flag and processcon > totalcon:
                all_sub = [p["pop"] for p in pops_sub]
                flag = False
                pop, fit = environmental_selection(self._merge(arch, *all_sub, pop), self.n_pop, processcon, totalcon)

            # Reproduction
            if self.operator_type == 1:
                mating = tournament_selection(2, 2 * self.n_pop, fit, self.rng)
                p1_idx = mating[: self.n_pop]
                p2_idx = mating[self.n_pop :]
                off_x = operator_de(pop.x, pop.x[p1_idx], pop.x[p2_idx], lo, hi, self.rng)
            else:
                mating = tournament_selection(2, self.n_pop, fit, self.rng)
                off_x = operator_ga(pop.x[mating], lo, hi, self.rng)

            offspring = self._evaluate(problem, off_x)

            if flag:
                for p_dict in pops_sub:
                    c_idx = p_dict["idx"]
                    p_dict["pop"], p_dict["fit"] = environmental_selection(
                        self._merge(p_dict["pop"], offspring), self.n_pop, c_idx, totalcon
                    )

            pop, fit = environmental_selection(self._merge(pop, offspring), self.n_pop, processcon, totalcon)
            obj_values.append(float(np.sum(np.abs(pop.f))))

            state = False
            if processcon <= totalcon:
                if len(obj_values) >= 2:
                    fit0 = cal_fitness(pop.f, pop.g, processcon, totalcon)
                    nc = int(np.sum(fit0 < 1.0))
                    max_change = abs(obj_values[-1] - obj_values[-2])
                    if nc >= self.n_pop:
                        thresh = self.change_threshold * abs((obj_values[-1] / self.n_pop) / problem.n_obj) * (10 ** (problem.n_obj - 2))
                        if max_change <= thresh:
                            state = True

            if state and processcon <= totalcon:
                ns = True
                all_sub = [p["pop"] for p in pops_sub]
                merged_sub = self._merge(*all_sub)
                fit_sub = cal_fitness(merged_sub.f, merged_sub.g, 0, totalcon)
                front_no = np.argsort(fit_sub)

                if processcon == 0:
                    ranks = []
                    for j in range(totalcon):
                        sub_fronts = front_no[j * self.n_pop : (j + 1) * self.n_pop]
                        ranks.append(int(np.min(sub_fronts)))
                    seq = list(np.argsort(ranks)[::-1])
                    index = 0
                else:
                    processed.append(processcon)
                    pops_sub[processcon - 1]["processed"] = 1
                    min_idx = int(np.min(front_no[(processcon - 1) * self.n_pop : processcon * self.n_pop]))
                    for i in range(totalcon):
                        if i != processcon - 1:
                            max_idx = int(np.max(front_no[i * self.n_pop : (i + 1) * self.n_pop]))
                            if max_idx <= min_idx:
                                pops_sub[i]["processed"] = 1

                unpro = sum(p["processed"] for p in pops_sub)
                if unpro < totalcon:
                    while index < len(seq) and pops_sub[seq[index]]["processed"] == 1:
                        index += 1
                    processcon = seq[index] + 1 if index < len(seq) else totalcon + 1
                else:
                    processcon = totalcon + 1

            if problem.eval_count >= problem.max_evals * 0.7:
                processcon = totalcon + 1

            if flag:
                arch = archive_selection(self._merge(arch, pop, offspring), self.n_pop)

            history["fe"].append(float(problem.eval_count))
            history["processcon"].append(float(processcon))

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
            final_nd_idx = np.flatnonzero(feas_mask)[nd_idx]
            nondom_pop = Population(x=pop.x[final_nd_idx], f=pop.f[final_nd_idx], cv=pop.cv[final_nd_idx])
        else:
            nondom_pop = Population(x=np.empty((0, len(lo))), f=np.empty((0, problem.n_obj)), cv=np.empty(0))

        return Result(population=pop, feasible_nondominated=nondom_pop, eval_count=problem.eval_count, history=history)
