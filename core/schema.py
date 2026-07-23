"""约束多目标优化问题和种群的数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """标准评估结果，包含目标函数值、约束矩阵与约束违反度。"""

    f: Array
    cv: Array
    g: Array | None = None
    h: Array | None = None


class CMOP(Protocol):
    """CMOP 的统一问题接口，所有目标默认按最小化处理。"""

    lower: Array
    upper: Array
    n_var: int
    n_obj: int
    max_evals: int

    @property
    def eval_count(self) -> int:
        """当前累计函数评估次数 (FE)。"""
        ...

    def evaluate(self, x: Array) -> EvaluationResult:
        """评估决策变量矩阵 x 并返回 EvaluationResult，同时累加评估次数。"""
        ...


@dataclass(frozen=True)
class Population:
    """一个种群的决策变量、目标值和约束违反度等公共属性。"""

    x: Array
    f: Array
    cv: Array
    g: Array | None = None
    h: Array | None = None


@dataclass(frozen=True)
class Result:
    """一次算法运行的最终种群、可行非支配解和演化记录。"""

    population: Population
    feasible_nondominated: Population
    eval_count: int
    history: dict[str, list[float]]
