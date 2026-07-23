"""POCEA (Paired Offspring Generation Based Constrained Evolutionary Algorithm) Python 实现。

对应 He 等人 2021 年 IEEE TEVC 论文及 PlatEMO 源码 POCEA.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.pocea.chp import chp
from algorithms.pocea.selection import association, reference_vector_adaptation, rvea_selection
from core.operators import operator_ga, uniform_point
from core.schema import CMOP, Array, Population, Result


class POCEA:
    """POCEA 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        k: int = 5,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.k = k
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

        # 1. 均匀生成参考向量
        v0, real_n = uniform_point(self.n_pop, m)
        self.n_pop = real_n
        vs0, l_len = uniform_point(math.floor(self.n_pop / self.k), m)

        v, vs = v0.copy(), vs0.copy()

        x_init = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        pop = self._evaluate(problem, x_init)

        history: dict[str, list[float]] = {"fe": []}

        while problem.eval_count < problem.max_evals:
            index_mat, theta_mat, dis_arr = association(pop, vs, self.k)
            cv = pop.cv
            rf = float(np.mean(cv < 1e-6)) if len(cv) > 0 else 1.0

            offspring_list: list[Population] = []

            for i in range(l_len):
                sub_indices = index_mat[:, i]
                sub_pop = pop.x[sub_indices]
                sub_cv = cv[sub_indices]
                sub_theta = theta_mat[sub_indices, i]

                if float(np.mean(sub_theta)) >= (np.pi / l_len / 2.0):
                    sorted_dis_idx = np.argsort(dis_arr)
                    selected = sorted_dis_idx[: min(self.k, len(sorted_dis_idx))]
                    all_sub_idx = np.unique(np.concatenate([sub_indices, selected]))
                    var_epsilon = float(np.max(cv[all_sub_idx])) if len(all_sub_idx) > 0 else 0.0
                    sub_pop_merged = pop.x[all_sub_idx]
                    sub_cv_merged = cv[all_sub_idx]
                else:
                    var_epsilon = float(np.min(sub_cv)) * (1.0 - rf) + float(np.mean(sub_cv)) * rf
                    sub_pop_merged = sub_pop
                    sub_cv_merged = sub_cv

                if len(sub_pop_merged) < 2:
                    p1_idx, p2_idx = 0, 0
                else:
                    perm = self.rng.permutation(len(sub_pop_merged))
                    p1_idx, p2_idx = perm[0], perm[1]

                winner_idx, loser_idx = chp(sub_cv_merged[p1_idx], sub_cv_merged[p2_idx], var_epsilon)
                p_win = sub_pop_merged[winner_idx : winner_idx + 1]
                p_lose = sub_pop_merged[loser_idx : loser_idx + 1]

                # 交叉配对生成子代
                parents_pair = np.vstack([p_lose, p_win])
                off_x = operator_ga(parents_pair, lo, hi, self.rng)
                off = self._evaluate(problem, off_x)
                offspring_list.append(off)

            offsprings = self._merge(*offspring_list)

            # RVEA 选择
            fe_ratio = problem.eval_count / problem.max_evals
            pop = rvea_selection(self._merge(pop, offsprings), v, self.n_pop, fe_ratio)

            # 定期调节参考向量
            curr_gen = math.ceil(problem.eval_count / self.n_pop)
            check_gen = math.ceil(0.1 * problem.max_evals / self.n_pop)
            if check_gen > 0 and curr_gen % check_gen == 0:
                v, vs = reference_vector_adaptation(pop.f, v0, vs0)

            history["fe"].append(float(problem.eval_count))

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
            nondom_pop = Population(x=np.empty((0, len(lo))), f=np.empty((0, m)), cv=np.empty(0))

        return Result(population=pop, feasible_nondominated=nondom_pop, eval_count=problem.eval_count, history=history)
