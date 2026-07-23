"""IM-C-MOEA-D (Inverse Modeling Constrained MOEA/D) Python 实现。

对应 Farias & Araujo 2024 年 IEEE SMC 论文及 PlatEMO 源码 IMCMOEAD.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.im_c_moea_d.constraint import apply_constraint_handling
from algorithms.im_c_moea_d.operators import kmeans_clusters
from algorithms.im_c_moea_d.replacement import global_replacement
from core.operators import operator_ga, tournament_selection, uniform_point
from core.schema import CMOP, Array, Population, Result


class IMCMOEAD:
    """IM-C-MOEA-D 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        n_clusters: int = 10,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.k_clusters = n_clusters
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
        t_size = math.ceil(self.n_pop / 10.0)

        dist_w = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        b_neighbors = np.argsort(dist_w, axis=1)[:, :t_size]

        # 2. 初始化种群
        x_init = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        pop = self._evaluate(problem, x_init)

        ideal_point = np.min(pop.f, axis=0)
        history: dict[str, list[float]] = {"fe": []}

        while problem.eval_count < problem.max_evals:
            labels = kmeans_clusters(pop.f, k=min(self.k_clusters, self.n_pop), rng=self.rng)

            offspring_list: list[Population] = []
            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_idx = np.flatnonzero(labels == label)
                cluster_size = len(cluster_idx)

                # 簇内二元锦标赛选择
                mating = tournament_selection(2, cluster_size, pop.cv[cluster_idx], self.rng)
                parents_idx = cluster_idx[mating]

                off_x = operator_ga(pop.x[parents_idx], lo, hi, self.rng)
                off = self._evaluate(problem, off_x)
                offspring_list.append(off)

            offsprings = self._merge(*offspring_list)

            # 约束惩罚转换目标空间
            pop_obj_mod = apply_constraint_handling(pop)
            off_obj_mod = apply_constraint_handling(offsprings)

            # 更新理想点与 Nadir 点
            combined_mod = np.vstack([pop_obj_mod, off_obj_mod])
            ideal_point = np.minimum(ideal_point, np.min(combined_mod, axis=0))
            nadir_point = np.maximum(np.max(combined_mod, axis=0), ideal_point + 1e-6)

            span = nadir_point - ideal_point + 1e-15
            pop_norm = (pop_obj_mod - ideal_point) / span
            off_norm = (off_obj_mod - ideal_point) / span

            # 全局替换更新种群
            pop = global_replacement(pop, offsprings, weights, b_neighbors, t_size, pop_norm, off_norm)
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
