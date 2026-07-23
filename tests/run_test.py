"""通用算法单机测试与演示脚本 (不保存文件结果)。

支持命令行参数动态指定算法 (--algorithms / --algorithm) 和 Benchmark 问题 (--problems / --problem)。
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from core.metrics import calculate_hv, calculate_igd
from core.problem import PymooProblemAdapter
from experiments.config import ALGORITHM_REGISTRY, PROBLEM_REGISTRY
from experiments.run_experiment import get_reference_front_and_point


def main() -> None:
    """运行多算法/多 benchmark 测试。"""
    parser = argparse.ArgumentParser(
        description="通用算法与 Benchmark 快速测试脚本 (不落盘结果)"
    )
    parser.add_argument(
        "--algorithms",
        "--algorithm",
        nargs="+",
        default=["DSOCOL"],
        help=f"选择算法，支持: {list(ALGORITHM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--problems",
        "--problem",
        nargs="+",
        default=["C1DTLZ1"],
        help=f"选择 Benchmark 问题，支持: {list(PROBLEM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--population-size", "--pop-size", type=int, default=100, help="种群规模"
    )
    parser.add_argument(
        "--max-evals", type=int, default=30000, help="最大函数评估次数 FEmax"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 校验输入
    for algo_name in args.algorithms:
        if algo_name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"未知算法 '{algo_name}'，可选算法: {list(ALGORITHM_REGISTRY.keys())}"
            )
    for prob_name in args.problems:
        if prob_name not in PROBLEM_REGISTRY:
            raise ValueError(
                f"未知问题 '{prob_name}'，可选问题: {list(PROBLEM_REGISTRY.keys())}"
            )

    print("==================================================")
    print(f"开始算法测试: 算法={args.algorithms}, 问题={args.problems}")
    print(f"参数: 种群规模={args.population_size}, FEmax={args.max_evals}, Seed={args.seed}")
    print("==================================================")

    for prob_name in args.problems:
        for algo_name in args.algorithms:
            print(f"\n---> 运行 [{algo_name}] 在 [{prob_name}] 上 (Seed={args.seed})...")
            prob_cls = PROBLEM_REGISTRY[prob_name]
            algo_cls = ALGORITHM_REGISTRY[algo_name]

            raw_prob = prob_cls()
            problem = PymooProblemAdapter(raw_prob, max_evals=args.max_evals)
            algorithm = algo_cls(population_size=args.population_size, seed=args.seed)

            ref_front, ref_point = get_reference_front_and_point(raw_prob, problem)

            t0 = time.perf_counter()
            result = algorithm.run(problem)
            elapsed = time.perf_counter() - t0

            feas_pop = result.feasible_nondominated
            n_feasible = len(feas_pop.x)

            igd_val = (
                calculate_igd(feas_pop.f, ref_front)
                if ref_front is not None and len(ref_front) > 0
                else float("nan")
            )
            hv_val = calculate_hv(feas_pop.f, ref_point)

            print("--------------------------------------------------")
            print(f"[{algo_name} @ {prob_name}] 运行完成 (耗时: {elapsed:.2f} s)")
            print(f"  总 FE: {result.eval_count} / {args.max_evals}")
            print(f"  最终主种群规模: {len(result.population.x)}")
            print(
                f"  主种群最终 CV [min, mean, max]: "
                f"[{result.population.cv.min():.4f}, {result.population.cv.mean():.4f}, {result.population.cv.max():.4f}]"
            )
            print(f"  可行非支配解数量: {n_feasible}")

            if n_feasible > 0:
                print(f"  IGD 指标: {igd_val:.4e}" if not np.isnan(igd_val) else "  IGD 指标: N/A")
                print(f"  HV  指标: {hv_val:.4f}")
                print("  部分可行非支配解的目标函数值 (前 5 个):")
                print(feas_pop.f[:5])

            if hasattr(result, "history") and result.history:
                if "epsilon" in result.history and len(result.history["epsilon"]) > 0:
                    print("  后期 epsilon 变化 (最后 5 代):", [round(e, 4) for e in result.history["epsilon"][-5:]])
                if "feasible_ratio_s1" in result.history and len(result.history["feasible_ratio_s1"]) > 0:
                    print("  后期 S1 可行率 (最后 5 代):", [round(r, 4) for r in result.history["feasible_ratio_s1"][-5:]])

            print("--------------------------------------------------")

    print("\n==================================================")
    print("所有算法与 Benchmark 测试完成 (未保存结果文件)。")
    print("==================================================")


if __name__ == "__main__":
    main()
