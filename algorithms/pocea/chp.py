"""POCEA 约束处理协议模块 (对应 CHP.m)。"""
from __future__ import annotations


def chp(sol1_cv: float, sol2_cv: float, var_epsilon: float) -> tuple[int, int]:
    """CHP (Constraint Handling Protocol) 确定 loser 和 winner (对应 CHP.m)。"""
    if sol1_cv <= var_epsilon and sol2_cv <= var_epsilon:
        return (0, 1) if sol1_cv <= sol2_cv else (1, 0)
    elif sol1_cv == sol2_cv:
        return (0, 1)
    else:
        return (0, 1) if sol1_cv < sol2_cv else (1, 0)
