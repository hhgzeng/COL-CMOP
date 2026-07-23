"""DVCEA 差分演化子代生成模块 (对应 DEgenerator_better.m, OperatorDE_pbest_1.m)。"""
from __future__ import annotations

import numpy as np
from core.operators import polynomial_mutation
from core.schema import CMOP, Array, Population


def operator_de_pbest_1_main(
    pop: Population, problem: CMOP, fit: Array, fea_indices: np.ndarray, p_ratio: float, rng: np.random.Generator
) -> Population:
    """对应 OperatorDE_pbest_1_main.m。"""
    n, d = pop.x.shape
    p_num = max(1, int(p_ratio * n))
    pbest_order = np.argsort(fit)[:p_num]

    off_x = pop.x.copy()
    for i in range(n):
        pbest_idx = rng.choice(pbest_order)
        r1, r2 = rng.choice(n, size=2, replace=False)
        x_i = pop.x[i]
        x_pb = pop.x[pbest_idx]
        x_r1 = pop.x[r1]
        x_r2 = pop.x[r2]

        v_sub = x_i[fea_indices] + 0.5 * (x_pb[fea_indices] - x_i[fea_indices]) + 0.5 * (x_r1[fea_indices] - x_r2[fea_indices])
        off_x[i, fea_indices] = v_sub

    mutated_x = polynomial_mutation(off_x, problem.lower, problem.upper, rng)
    res = problem.evaluate(mutated_x)
    return Population(x=np.asarray(mutated_x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)


def degenerator_better(
    pop: Population, problem: CMOP, infea_indices: np.ndarray, var_epsilon: float, rng: np.random.Generator
) -> Population:
    """对应 DEgenerator_better.m。"""
    n, d = pop.x.shape
    off_x = pop.x.copy()
    for i in range(n):
        r1, r2, r3 = rng.choice(n, size=3, replace=False)
        v_sub = pop.x[r1, infea_indices] + 0.5 * (pop.x[r2, infea_indices] - pop.x[r3, infea_indices])
        off_x[i, infea_indices] = v_sub

    mutated_x = polynomial_mutation(off_x, problem.lower, problem.upper, rng)
    res = problem.evaluate(mutated_x)
    return Population(x=np.asarray(mutated_x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)
