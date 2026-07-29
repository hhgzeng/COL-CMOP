"""实验配置文件与算法/问题注册表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

from algorithms import (
    APSEA,
    C3M,
    CMOCSO,
    CMOEMT,
    DRLOSEMCMO,
    DSOCOL,
    DSOCOL1,
    DSOCOL2,
    DSOCOL3,
    DSOCOL4,
    DSOCOL5,
    DVCEA,
    IMCMOEAD,
    LCMEA,
    POCEA,
)
from problems.cdtlz import C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ4
from problems.dascmop import (
    DASCMOP1,
    DASCMOP2,
    DASCMOP3,
    DASCMOP4,
    DASCMOP5,
    DASCMOP6,
    DASCMOP7,
    DASCMOP8,
    DASCMOP9,
)
from problems.dcdtlz import (
    DC1DTLZ1,
    DC1DTLZ3,
    DC2DTLZ1,
    DC2DTLZ3,
    DC3DTLZ1,
    DC3DTLZ3,
)
from problems.lircmop import (
    LIRCMOP1,
    LIRCMOP2,
    LIRCMOP3,
    LIRCMOP4,
    LIRCMOP5,
    LIRCMOP6,
    LIRCMOP7,
    LIRCMOP8,
    LIRCMOP9,
    LIRCMOP10,
    LIRCMOP11,
    LIRCMOP12,
    LIRCMOP13,
    LIRCMOP14,
)

ALGORITHM_REGISTRY: dict[str, Type[Any]] = {
    "DSOCOL": DSOCOL,
    "DSOCOL1": DSOCOL1,
    "DSOCOL2": DSOCOL2,
    "DSOCOL3": DSOCOL3,
    "DSOCOL4": DSOCOL4,
    "DSOCOL5": DSOCOL5,
    "APSEA": APSEA,
    "C3M": C3M,
    "CMOCSO": CMOCSO,
    "CMOEMT": CMOEMT,
    "DRLOSEMCMO": DRLOSEMCMO,
    "DVCEA": DVCEA,
    "IMCMOEAD": IMCMOEAD,
    "LCMEA": LCMEA,
    "POCEA": POCEA,
}


PROBLEM_REGISTRY: dict[str, Callable[..., Any]] = {
    # C-DTLZ Benchmarks
    "C1DTLZ1": C1DTLZ1,
    "C1DTLZ3": C1DTLZ3,
    "C2DTLZ2": C2DTLZ2,
    "C3DTLZ4": C3DTLZ4,
    # DC-DTLZ Benchmarks
    "DC1DTLZ1": DC1DTLZ1,
    "DC1DTLZ3": DC1DTLZ3,
    "DC2DTLZ1": DC2DTLZ1,
    "DC2DTLZ3": DC2DTLZ3,
    "DC3DTLZ1": DC3DTLZ1,
    "DC3DTLZ3": DC3DTLZ3,
    # DASCMOP Benchmarks (default difficulty=1)
    "DASCMOP1": lambda difficulty=1: DASCMOP1(difficulty=difficulty),
    "DASCMOP2": lambda difficulty=1: DASCMOP2(difficulty=difficulty),
    "DASCMOP3": lambda difficulty=1: DASCMOP3(difficulty=difficulty),
    "DASCMOP4": lambda difficulty=1: DASCMOP4(difficulty=difficulty),
    "DASCMOP5": lambda difficulty=1: DASCMOP5(difficulty=difficulty),
    "DASCMOP6": lambda difficulty=1: DASCMOP6(difficulty=difficulty),
    "DASCMOP7": lambda difficulty=1: DASCMOP7(difficulty=difficulty),
    "DASCMOP8": lambda difficulty=1: DASCMOP8(difficulty=difficulty),
    "DASCMOP9": lambda difficulty=1: DASCMOP9(difficulty=difficulty),
    # LIR-CMOP Benchmarks
    "LIRCMOP1": LIRCMOP1,
    "LIRCMOP2": LIRCMOP2,
    "LIRCMOP3": LIRCMOP3,
    "LIRCMOP4": LIRCMOP4,
    "LIRCMOP5": LIRCMOP5,
    "LIRCMOP6": LIRCMOP6,
    "LIRCMOP7": LIRCMOP7,
    "LIRCMOP8": LIRCMOP8,
    "LIRCMOP9": LIRCMOP9,
    "LIRCMOP10": LIRCMOP10,
    "LIRCMOP11": LIRCMOP11,
    "LIRCMOP12": LIRCMOP12,
    "LIRCMOP13": LIRCMOP13,
    "LIRCMOP14": LIRCMOP14,
}

BENCHMARK_CATEGORIES: dict[str, list[str]] = {
    "C-DTLZs": ["C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C3DTLZ4"],
    "DC-DTLZs": ["DC1DTLZ1", "DC1DTLZ3", "DC2DTLZ1", "DC2DTLZ3", "DC3DTLZ1", "DC3DTLZ3"],
    "DAS-CMOP": [f"DASCMOP{i}" for i in range(1, 10)],
    "LIR-CMOP": [f"LIRCMOP{i}" for i in range(1, 15)],
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
