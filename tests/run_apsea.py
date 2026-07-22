"""APSEA 算法在 C1DTLZ1 问题上的运行示例脚本。"""

from __future__ import annotations

import argparse
import numpy as np

from algorithms.apsea import APSEA
from core.problem import PymooProblemAdapter
from problems.cdtlz import C1DTLZ1


def main() -> None:
    """运行一个可复现的 APSEA 示例实验。"""
    parser = argparse.ArgumentParser(description="运行 APSEA 示例实验")
    parser.add_argument(
        "--population-size", type=int, default=100, help="主群体 Population1 的规模"
    )
    parser.add_argument("--max-evals", type=int, default=30000, help="最大函数评估次数 FEmax")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    problem = PymooProblemAdapter(C1DTLZ1(), max_evals=args.max_evals)
    algorithm = APSEA(
        population_size=args.population_size,
        seed=args.seed,
    )
    result = algorithm.run(problem)
    print(f"==================================================")
    print(f"APSEA 评估完成！总 FE：{result.eval_count} / {args.max_evals}")
    print(f"最终主群体规模：{len(result.population.x)}")
    print(f"主群体最终约束违反度 CV [min, mean, max]: [{result.population.cv.min():.4f}, {result.population.cv.mean():.4f}, {result.population.cv.max():.4f}]")
    print(f"可行非支配解数量：{len(result.feasible_nondominated.x)}")
    if len(result.feasible_nondominated.x) > 0:
        print("部分可行非支配解的目标函数值 (前 5 个)：")
        print(result.feasible_nondominated.f[:5])
    print(f"==================================================")


if __name__ == "__main__":
    main()
