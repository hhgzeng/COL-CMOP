"""CMOCSO 遗传与粒子群算子模块 (对应 CompetitiveOperator.m 与 CooperativeOperator.m)。"""
from __future__ import annotations

import numpy as np

from core.operators import operator_ga, polynomial_mutation
from core.schema import CMOP, Array, Population


def competitive_operator(
    problem: CMOP, loser_x: Array, winner_x: Array, y_val: float, rng: np.random.Generator
) -> Population:
    """竞争算子 (对应 CompetitiveOperator.m)。"""
    n, d = loser_x.shape
    loser_vel = np.zeros((n, d), dtype=float)

    r1 = rng.random((n, 1))
    r2 = rng.random((n, 1))

    off_vel = r1 * loser_vel + r2 * (winner_x - loser_x) * y_val
    n_dir = rng.integers(1, 3)
    off_x = loser_x + off_vel + r1 * (off_vel - loser_vel) * ((-1) ** n_dir)

    concat_x = np.vstack([off_x, winner_x])
    mutated_x = polynomial_mutation(concat_x, problem.lower, problem.upper, rng)
    res = problem.evaluate(mutated_x)
    return Population(x=np.asarray(mutated_x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)


def cooperative_operator(problem: CMOP, parent_x: Array, rng: np.random.Generator) -> Population:
    """协同算子 (对应 CooperativeOperator.m)。"""
    off_x = operator_ga(parent_x, problem.lower, problem.upper, rng)
    res = problem.evaluate(off_x)
    return Population(x=np.asarray(off_x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)
