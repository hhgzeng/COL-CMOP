"""实验配置文件与算法/问题注册表。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from algorithms.apsea import APSEA
from algorithms.dsocol import DSOCOL
from problems.cdtlz import C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4

ALGORITHM_REGISTRY: dict[str, Type] = {
    "DSOCOL": DSOCOL,
    "APSEA": APSEA,
}

PROBLEM_REGISTRY: dict[str, Type] = {
    "C1DTLZ1": C1DTLZ1,
    "C1DTLZ3": C1DTLZ3,
    "C2DTLZ2": C2DTLZ2,
    "C3DTLZ1": C3DTLZ1,
    "C3DTLZ4": C3DTLZ4,
}


@dataclass
class ExperimentConfig:
    """标准实验配置对象。"""

    population_size: int = 100
    max_evals: int = 100000
    n_runs: int = 30
    base_seed: int = 42
    algorithms: list[str] = field(default_factory=lambda: ["DSOCOL", "APSEA"])
    problems: list[str] = field(default_factory=lambda: ["C1DTLZ1"])
    results_dir: str = "results"
