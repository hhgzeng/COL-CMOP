"""实验主程序：支持单次运行、批量 independent runs 及结果自动落盘 (CSV/NPZ)。"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.metrics import calculate_hv, calculate_igd
from core.problem import PymooProblemAdapter
from experiments.config import ALGORITHM_REGISTRY, PROBLEM_REGISTRY, ExperimentConfig


def get_reference_front_and_point(
    problem_instance, adapter: PymooProblemAdapter
) -> tuple[np.ndarray | None, np.ndarray]:
    """获取问题对应的参考 Pareto Front 和超体积 Reference Point。"""
    ref_front = None
    try:
        ref_front = problem_instance.pareto_front()
    except Exception:
        ref_front = None

    if ref_front is not None and len(ref_front) > 0:
        nadir = np.max(ref_front, axis=0)
        ref_point = nadir * 1.1 + 0.1
    else:
        # 默认回退参考点
        ref_point = np.full(adapter.n_obj, 2.0)

    return ref_front, ref_point


def run_single_run(
    algo_name: str,
    prob_name: str,
    seed: int,
    max_evals: int = 100000,
    population_size: int = 100,
    save_dir: str | Path = "results",
) -> dict[str, float]:
    """执行单个 Seed 的算法运行，记录耗时与指标并落盘 NPZ 文件。"""
    save_dir = Path(save_dir) / algo_name / prob_name
    save_dir.mkdir(parents=True, exist_ok=True)

    prob_cls = PROBLEM_REGISTRY[prob_name]
    algo_cls = ALGORITHM_REGISTRY[algo_name]

    raw_prob = prob_cls()
    adapter = PymooProblemAdapter(raw_prob, max_evals=max_evals)
    algo = algo_cls(population_size=population_size, seed=seed)

    ref_front, ref_point = get_reference_front_and_point(raw_prob, adapter)

    start_time = time.perf_counter()
    result = algo.run(adapter)
    elapsed_time = time.perf_counter() - start_time

    feas_pop = result.feasible_nondominated
    n_feasible = len(feas_pop.x)

    igd_val = (
        calculate_igd(feas_pop.f, ref_front) if ref_front is not None else float("nan")
    )
    hv_val = calculate_hv(feas_pop.f, ref_point)

    # 保存单次原始数据 NPZ
    npz_path = save_dir / f"run_seed_{seed}.npz"
    np.savez_compressed(
        npz_path,
        x=result.population.x,
        f=result.population.f,
        cv=result.population.cv,
        g=result.population.g if result.population.g is not None else np.empty(0),
        h=result.population.h if result.population.h is not None else np.empty(0),
        feas_x=feas_pop.x,
        feas_f=feas_pop.f,
        feas_cv=feas_pop.cv,
        eval_count=result.eval_count,
        elapsed_time=elapsed_time,
        igd=igd_val,
        hv=hv_val,
        n_feasible=n_feasible,
    )

    return {
        "algorithm": algo_name,
        "problem": prob_name,
        "seed": seed,
        "eval_count": result.eval_count,
        "elapsed_time": elapsed_time,
        "n_feasible_nd": n_feasible,
        "igd": igd_val,
        "hv": hv_val,
    }


def run_batch_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """按配置进行多算法、多问题、N 次独立重复实验并生成汇总 CSV/NPZ 报告。"""
    all_metrics: list[dict[str, float]] = []

    print(f"==================================================")
    print(f"开始执行实验批次: 算法={config.algorithms}, 问题={config.problems}")
    print(
        f"独立重复次数: {config.n_runs}, 最大 FE: {config.max_evals}, 种群规模: {config.population_size}"
    )
    print(f"==================================================")

    for prob_name in config.problems:
        for algo_name in config.algorithms:
            print(
                f"\n---> 运行 [{algo_name}] 在 [{prob_name}] 上 ({config.n_runs} 次独立运行)..."
            )
            for run_idx in range(config.n_runs):
                seed = config.base_seed + run_idx * 100
                metrics = run_single_run(
                    algo_name=algo_name,
                    prob_name=prob_name,
                    seed=seed,
                    max_evals=config.max_evals,
                    population_size=config.population_size,
                    save_dir=config.results_dir,
                )
                all_metrics.append(metrics)
                print(
                    f"  [Run {run_idx + 1:02d}/{config.n_runs:02d} | Seed {seed}] "
                    f"FE={metrics['eval_count']} | 可行解={metrics['n_feasible_nd']} | "
                    f"IGD={metrics['igd']:.4e} | HV={metrics['hv']:.4f} | 时间={metrics['elapsed_time']:.2f}s"
                )

    df_detail = pd.DataFrame(all_metrics)

    # 导出明细数据 CSV
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_detail.to_csv(out_dir / "detailed_runs.csv", index=False)

    # 导出统计汇总 CSV (mean ± std)
    summary_rows = []
    for (algo, prob), group in df_detail.groupby(["algorithm", "problem"]):
        summary_rows.append(
            {
                "Algorithm": algo,
                "Problem": prob,
                "Runs": len(group),
                "Feasible_Ratio": f"{np.mean(group['n_feasible_nd'] > 0):.2%}",
                "Mean_Feasible_ND": f"{group['n_feasible_nd'].mean():.2f} ± {group['n_feasible_nd'].std():.2f}",
                "IGD_Mean": group["igd"].mean(),
                "IGD_Std": group["igd"].std(),
                "HV_Mean": group["hv"].mean(),
                "HV_Std": group["hv"].std(),
                "Time_Mean_Sec": group["elapsed_time"].mean(),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    print("\n==================================================")
    print(f"实验全部完成！统计汇总已存入 {out_dir / 'summary_metrics.csv'}")
    print(df_summary.to_string(index=False))
    print("==================================================")

    return df_summary


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="CMOP 批量实验脚本")
    parser.add_argument(
        "--algorithms", nargs="+", default=["DSOCOL", "APSEA"], help="包含的算法列表"
    )
    parser.add_argument(
        "--problems", nargs="+", default=["C1DTLZ1"], help="包含的问题列表"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="独立重复运行次数 (默认 5 次演示，论文标准 30 次)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=20000,
        help="最大 FE 预算 (默认 20000 演示，论文标准 100000)",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="种群规模")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="结果输出目录"
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        population_size=args.pop_size,
        max_evals=args.max_evals,
        n_runs=args.n_runs,
        algorithms=args.algorithms,
        problems=args.problems,
        results_dir=args.results_dir,
    )
    run_batch_experiment(config)


if __name__ == "__main__":
    main()
