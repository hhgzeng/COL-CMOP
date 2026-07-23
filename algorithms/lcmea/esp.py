"""LCMEA 采样子代生成模块 (对应 ESP.m)。"""
from __future__ import annotations

import numpy as np
from core.operators import operator_ga
from core.schema import CMOP, Population


def esp_offspring_generation(pop: Population, problem: CMOP, rng: np.random.Generator) -> Population:
    """ESP (Efficient Sampling Approach) 采样算子生成子代 (对应 ESP.m)。"""
    n, d = pop.x.shape
    lo, hi = problem.lower, problem.upper

    mating1 = rng.choice(n, size=n, replace=True)
    mating2 = rng.choice(n, size=n, replace=True)

    off_x = pop.x.copy()
    for i in range(n):
        p1 = pop.x[mating1[i]]
        p2 = pop.x[mating2[i]]
        direction = p1 - p2
        step = rng.normal(0, 0.5, size=d) * direction
        off_x[i] = np.clip(pop.x[i] + step, lo, hi)

    off_x = operator_ga(off_x, lo, hi, rng)
    res = problem.evaluate(off_x)
    return Population(x=np.asarray(off_x, dtype=float), f=res.f, cv=res.cv, g=res.g, h=res.h)
