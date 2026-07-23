"""DRLOS-EMCMO (EMCMO with Deep Reinforcement Learning-assisted Operator Selection) Python 实现。

对应 Ming 等人 2024 年 IEEE/CAA JAS 论文及 PlatEMO 源码 DRLOSEMCMO.m。
"""
from __future__ import annotations

import math
import numpy as np

from algorithms.drlos_emcmo.dropout import SimpleQNet
from algorithms.drlos_emcmo.fitness import cal_fitness
from algorithms.drlos_emcmo.selection import environmental_selection
from core.operators import operator_de, operator_ga, tournament_selection
from core.schema import CMOP, Array, Population, Result


class DRLOSEMCMO:
    """DRLOS-EMCMO 算法主类。"""

    def __init__(
        self,
        population_size: int = 100,
        greedy: float = 0.95,
        gamma: float = 0.9,
        seed: int | None = None,
    ):
        self.n_pop = population_size
        self.greedy = greedy
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, problem: CMOP, x: Array) -> Population:
        res = problem.evaluate(x)
        return Population(x=np.asarray(x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)

    @staticmethod
    def _take(pop: Population, idx: Array) -> Population:
        return Population(
            x=pop.x[idx],
            f=pop.f[idx],
            cv=pop.cv[idx],
            g=pop.g[idx] if pop.g is not None else None,
            h=pop.h[idx] if pop.h is not None else None,
        )

    @staticmethod
    def _merge(*pops: Population) -> Population:
        x_concat = np.concatenate([p.x for p in pops if len(p.x) > 0], axis=0)
        f_concat = np.concatenate([p.f for p in pops if len(p.f) > 0], axis=0)
        cv_concat = np.concatenate([p.cv for p in pops if len(p.cv) > 0], axis=0)
        total_len = len(x_concat)
        if all(p.g is not None for p in pops if len(p.x) > 0):
            g_concat = np.concatenate([p.g for p in pops if len(p.x) > 0], axis=0)
            if len(g_concat) != total_len:
                g_concat = None
        else:
            g_concat = None

        if all(p.h is not None for p in pops if len(p.x) > 0):
            h_concat = np.concatenate([p.h for p in pops if len(p.x) > 0], axis=0)
            if len(h_concat) != total_len:
                h_concat = None
        else:
            h_concat = None

        return Population(x=x_concat, f=f_concat, cv=cv_concat, g=g_concat, h=h_concat)

    def run(self, problem: CMOP) -> Result:
        lo, hi = problem.lower, problem.upper
        x1 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))
        x2 = self.rng.uniform(lo, hi, size=(self.n_pop, len(lo)))

        pop1 = self._evaluate(problem, x1)
        pop2 = self._evaluate(problem, x2)

        fit1 = cal_fitness(pop1.f, pop1.cv)
        fit2 = cal_fitness(pop2.f, None)

        transfer_state = 0
        cnt = 0
        data: list[list[float]] = []
        model_built = False
        count_gen = 0
        net = SimpleQNet(in_features=4, rng=self.rng)

        history: dict[str, list[float]] = {"fe": [], "operator": []}

        while problem.eval_count < problem.max_evals:
            gen = math.ceil(problem.eval_count / (2 * self.n_pop))

            # 特征提取
            avg_f = float(np.mean(np.sum(pop1.f, axis=1)))
            avg_cv = float(np.mean(pop1.cv))
            f_max = np.max(pop1.f, axis=0)
            f_min = np.min(pop1.f, axis=0)
            avg_d = float(np.sum(f_max - f_min))

            # 动作/算子选择 (1: GA, 2: DE)
            if gen <= 200:
                operator = int(self.rng.integers(1, 3))
            else:
                if not model_built:
                    model_built = True
                    operator = int(self.rng.integers(1, 3))
                else:
                    if self.rng.random() > self.greedy:
                        operator = int(self.rng.integers(1, 3))
                    else:
                        s1 = np.array([[avg_f, avg_cv, avg_d, 1.0]])
                        s2 = np.array([[avg_f, avg_cv, avg_d, 2.0]])
                        r1 = net.predict(s1)[0]
                        r2 = net.predict(s2)[0]
                        operator = 1 if r1 >= r2 else 2

            cnt += 1
            val_off1: Population
            val_off2: Population

            if transfer_state == 0:
                if operator == 1:
                    idx1 = self.rng.integers(0, self.n_pop, size=self.n_pop)
                    idx2 = self.rng.integers(0, self.n_pop, size=self.n_pop)
                    off1_x = operator_ga(pop1.x[idx1], lo, hi, self.rng)
                    off2_x = operator_ga(pop2.x[idx2], lo, hi, self.rng)
                else:
                    idx1_p1 = self.rng.integers(0, self.n_pop, size=self.n_pop)
                    idx1_p2 = self.rng.integers(0, self.n_pop, size=self.n_pop)
                    idx2_p1 = self.rng.integers(0, self.n_pop, size=self.n_pop)
                    idx2_p2 = self.rng.integers(0, self.n_pop, size=self.n_pop)
                    off1_x = operator_de(pop1.x, pop1.x[idx1_p1], pop1.x[idx1_p2], lo, hi, self.rng)
                    off2_x = operator_de(pop2.x, pop2.x[idx2_p1], pop2.x[idx2_p2], lo, hi, self.rng)

                val_off1 = self._evaluate(problem, off1_x)
                val_off2 = self._evaluate(problem, off2_x)

                pop1, fit1, _ = environmental_selection(self._merge(pop1, val_off1, val_off2), self.n_pop, mode=1)
                pop2, fit2, _ = environmental_selection(self._merge(pop2, val_off2, val_off1), self.n_pop, mode=2)

                if problem.eval_count / problem.max_evals >= 0.2:
                    transfer_state = 1
            else:
                if operator == 1:
                    m1 = tournament_selection(2, self.n_pop, fit1, self.rng)
                    m2 = tournament_selection(2, self.n_pop, fit2, self.rng)
                    off1_x = operator_ga(pop1.x[m1], lo, hi, self.rng)
                    off2_x = operator_ga(pop2.x[m2], lo, hi, self.rng)
                else:
                    m1 = tournament_selection(2, 2 * self.n_pop, fit1, self.rng)
                    m2 = tournament_selection(2, 2 * self.n_pop, fit2, self.rng)
                    off1_x = operator_de(pop1.x, pop1.x[m1[: self.n_pop]], pop1.x[m1[self.n_pop :]], lo, hi, self.rng)
                    off2_x = operator_de(pop2.x, pop2.x[m2[: self.n_pop]], pop2.x[m2[self.n_pop :]], lo, hi, self.rng)

                val_off1 = self._evaluate(problem, off1_x)
                val_off2 = self._evaluate(problem, off2_x)

                _, _, next2 = environmental_selection(self._merge(pop2, val_off2), self.n_pop, mode=1)
                succ_rate1 = (np.sum(next2[: self.n_pop]) / 100.0) - (np.sum(next2[self.n_pop :]) / 50.0)

                _, _, next1 = environmental_selection(self._merge(pop1, val_off1), self.n_pop, mode=2)
                succ_rate2 = (np.sum(next1[: self.n_pop]) / 100.0) - (np.sum(next1[self.n_pop :]) / 50.0)

                rand_perm1 = self.rng.permutation(self.n_pop)[: self.n_pop // 2]
                rand_perm2 = self.rng.permutation(self.n_pop)[: self.n_pop // 2]

                if succ_rate1 > 0:
                    pop1, fit1, _ = environmental_selection(self._merge(pop1, val_off1, self._take(pop2, rand_perm1)), self.n_pop, mode=1)
                else:
                    pop1, fit1, _ = environmental_selection(self._merge(pop1, val_off1, val_off2), self.n_pop, mode=1)

                if succ_rate2 > 0:
                    pop2, fit2, _ = environmental_selection(self._merge(pop2, val_off2, self._take(pop1, rand_perm2)), self.n_pop, mode=2)
                else:
                    pop2, fit2, _ = environmental_selection(self._merge(pop2, val_off2, val_off1), self.n_pop, mode=2)

            # 新状态记录与 Replay Buffer 更新
            avg_f1 = float(np.mean(np.sum(pop1.f, axis=1)))
            avg_cv1 = float(np.mean(pop1.cv))
            f_max1 = np.max(pop1.f, axis=0)
            f_min1 = np.min(pop1.f, axis=0)
            avg_d1 = float(np.sum(f_max1 - f_min1))

            reward = (avg_f1 + avg_cv1 + avg_d1) - (avg_f + avg_cv + avg_d)
            record = [avg_f, avg_cv, avg_d, float(operator), reward, avg_f1, avg_cv1, avg_d1]
            data.append(record)
            if len(data) > 500:
                data.pop(0)

            # 定期训练模型
            if model_built:
                count_gen += 1
                if count_gen > 50 and len(data) >= 20:
                    sample_size = min(200, len(data))
                    indices = self.rng.choice(len(data), size=sample_size, replace=False)
                    batch = np.array([data[idx] for idx in indices])
                    tr_x = batch[:, :4]
                    tr_y = batch[:, 4] + self.gamma * batch[:, 7]
                    net.fit(tr_x, tr_y, epochs=10)
                    count_gen = 0

            history["fe"].append(float(problem.eval_count))
            history["operator"].append(float(operator))

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
            nondom_pop = Population(x=np.empty((0, len(lo))), f=np.empty((0, problem.n_obj)), cv=np.empty(0))

        return Result(population=pop1, feasible_nondominated=nondom_pop, eval_count=problem.eval_count, history=history)
