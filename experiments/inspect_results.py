"""Inspect NPZ experiment results and optionally export a CSV summary.

Examples::

    python experiments/inspect_results.py \
        --results-dir results-exp --output-dir results-exp --export-csv
    python experiments/inspect_results.py \
        --results-dir results-ablation --algo DSOCOL --category C-DTLZs --detail

``--results-dir`` is the directory containing algorithm result folders.  The
optional ``--output-dir`` controls where the exported CSV is written; it may
be different from the input directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.analysis_utils import (  # noqa: E402
    BENCHMARK_CATEGORIES,
    DEFAULT_RESULTS_DIR,
    discover_algorithms,
    ensure_category_structure,
)


def inspect_single_npz(npz_path: Path) -> dict:
    """Read the useful metadata and metrics from one NPZ file."""
    with np.load(npz_path) as data:
        files = data.files
        return {
            "file_name": npz_path.name,
            "keys": files,
            "seed": npz_path.stem.split("_")[-1],
            "n_feasible": int(data["n_feasible"])
            if "n_feasible" in files
            else len(data.get("feas_f", [])),
            "igd": float(data["igd"]) if "igd" in files else float("nan"),
            "hv": float(data["hv"]) if "hv" in files else float("nan"),
            "eval_count": int(data["eval_count"]) if "eval_count" in files else 0,
            "elapsed_time": float(data["elapsed_time"])
            if "elapsed_time" in files
            else 0.0,
            "x_shape": data["x"].shape if "x" in files else None,
            "f_shape": data["f"].shape if "f" in files else None,
            "feas_f_shape": data["feas_f"].shape if "feas_f" in files else None,
        }


def inspect_category(
    algo_name: str,
    category: str,
    results_dir: Path,
    detail: bool = False,
) -> list[dict]:
    """Inspect all NPZ files for one algorithm and category."""
    category_dir = results_dir / algo_name / category
    records: list[dict] = []

    if not category_dir.exists():
        print(f"⚠️ 路径不存在: {category_dir}")
        return records

    problem_dirs = sorted(item for item in category_dir.iterdir() if item.is_dir())
    if not problem_dirs:
        print(f"ℹ️ 算法 [{algo_name}] 在 Benchmark 分类 [{category}] 下暂无运行结果文件。")
        return records

    print("\n" + "=" * 80)
    print(f"🔍 批量查看结果: 算法 [{algo_name}] | Benchmark 分类 [{category}]")
    print("=" * 80)

    for problem_dir in problem_dirs:
        npz_files = sorted(problem_dir.glob("*.npz"))
        if not npz_files:
            continue

        igds, hvs, feasible_counts, times = [], [], [], []
        if detail:
            print(f"\n📌 测试问题: {problem_dir.name} (共 {len(npz_files)} 次独立运行 NPZ 文件)")
            print(
                f"{'文件名':<20} | {'Seed':<6} | {'IGD':<10} | {'HV':<10} | "
                f"{'可行解数':<8} | {'耗时(s)':<8} | {'解矩阵 (x)':<12} | {'目标矩阵 (f)':<12}"
            )
            print("-" * 95)

        for npz_file in npz_files:
            info = inspect_single_npz(npz_file)
            igds.append(info["igd"])
            hvs.append(info["hv"])
            feasible_counts.append(info["n_feasible"])
            times.append(info["elapsed_time"])

            if detail:
                x_shape = str(info["x_shape"]) if info["x_shape"] else "N/A"
                f_shape = str(info["f_shape"]) if info["f_shape"] else "N/A"
                print(
                    f"{info['file_name']:<20} | {info['seed']:<6} | "
                    f"{info['igd']:<10.4e} | {info['hv']:<10.4f} | "
                    f"{info['n_feasible']:<8} | {info['elapsed_time']:<8.2f} | "
                    f"{x_shape:<12} | {f_shape:<12}"
                )

        records.append(
            {
                "Algorithm": algo_name,
                "Category": category,
                "Problem": problem_dir.name,
                "Runs": len(npz_files),
                "IGD_Mean": np.mean(igds),
                "IGD_Std": np.std(igds),
                "HV_Mean": np.mean(hvs),
                "HV_Std": np.std(hvs),
                "Feas_Mean": np.mean(feasible_counts),
                "Time_Mean": np.mean(times),
            }
        )

    if records:
        frame = pd.DataFrame(records)
        print(f"\n📊 [{algo_name} - {category}] 问题汇总统计摘要:")
        print(
            frame.to_string(
                index=False,
                formatters={
                    "IGD_Mean": "{:.4e}".format,
                    "IGD_Std": "{:.4e}".format,
                    "HV_Mean": "{:.4f}".format,
                    "HV_Std": "{:.4f}".format,
                    "Feas_Mean": "{:.1f}".format,
                    "Time_Mean": "{:.2f}s".format,
                },
            )
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量查看算法 NPZ 结果并导出统计摘要。"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="NPZ 结果目录（默认: results）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV 输出目录（默认与 --results-dir 相同）",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="指定算法；不指定时扫描结果目录下的所有算法",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=list(BENCHMARK_CATEGORIES),
        help="指定 Benchmark 分类",
    )
    parser.add_argument("--detail", action="store_true", help="打印每个 Seed 的详细信息")
    parser.add_argument("--export-csv", action="store_true", help="导出 CSV 汇总")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser()
    output_dir = (args.output_dir or results_dir).expanduser()
    ensure_category_structure(results_dir)

    algorithms = [args.algo] if args.algo else discover_algorithms(results_dir)
    categories = [args.category] if args.category else list(BENCHMARK_CATEGORIES)

    all_records: list[dict] = []
    for algorithm in algorithms:
        for category in categories:
            all_records.extend(
                inspect_category(algorithm, category, results_dir, detail=args.detail)
            )

    if args.export_csv and all_records:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "npz_batch_inspection_summary.csv"
        pd.DataFrame(all_records).to_csv(output_csv, index=False)
        print(f"\n✅ 批量查看统计汇总已保存至: {output_csv}")


if __name__ == "__main__":
    main()
