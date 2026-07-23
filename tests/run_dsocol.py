"""DSOCOL 算法在 C1DTLZ1 问题上的运行示例脚本。"""

from __future__ import annotations

import argparse

from algorithms.dsocol import DSOCOL
from core.problem import PymooProblemAdapter
from problems.cdtlz import C1DTLZ1


def main() -> None:
    """运行一个可复现的 DSOCOL 示例实验。"""
    parser = argparse.ArgumentParser(description="运行 DSOCOL 示例实验")
    parser.add_argument(
        "--population-size", type=int, default=100, help="每个群体的规模"
    )
    parser.add_argument("--max-evals", type=int, default=30000, help="最大函数评估次数 FEmax")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    problem = PymooProblemAdapter(C1DTLZ1(), max_evals=args.max_evals)
    algorithm = DSOCOL(
        population_size=args.population_size,
        seed=args.seed,
    )
    result = algorithm.run(problem)
    print("==================================================")
    print(f"评估完成！总 FE：{result.eval_count} / {args.max_evals}")
    print(f"最终 S1 种群规模：{len(result.population.x)}")
    print(f"S1 最终约束违反度 CV [min, mean, max]: [{result.population.cv.min():.4f}, {result.population.cv.mean():.4f}, {result.population.cv.max():.4f}]")
    print(f"可行非支配解数量：{len(result.feasible_nondominated.x)}")
    print("演化后期的 epsilon 变化 (最后 10 代)：", [round(e, 4) for e in result.history["epsilon"][-10:]])
    print("演化后期的 S1 可行率 (最后 10 代)：", [round(r, 4) for r in result.history["feasible_ratio_s1"][-10:]])
    print("==================================================")


if __name__ == "__main__":
    main()
