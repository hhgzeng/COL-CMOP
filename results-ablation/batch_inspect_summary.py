"""方法一：批量查看与解析每个算法在各 Benchmark 分类下的 NPZ 文件数据内容与指标汇总。

用法示例：
    # 自动建立四类 Benchmark 文件夹并查看 APSEA 在 C-DTLZs 分类下的所有 NPZ 文件
    python batch_inspect_summary.py --algo APSEA --category C-DTLZs

    # 查看所有算法、所有 Benchmark 分类的数据摘要并导出 CSV
    python batch_inspect_summary.py --export-csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_CATEGORIES = {
    "C-DTLZs": ["C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C3DTLZ4"],
    "DC-DTLZs": ["DC1DTLZ1", "DC1DTLZ3", "DC2DTLZ1", "DC2DTLZ3", "DC3DTLZ1", "DC3DTLZ3"],
    "DAS-CMOP": [f"DASCMOP{i}" for i in range(1, 10)],
    "LIR-CMOP": [f"LIRCMOP{i}" for i in range(1, 15)],
}

RESULTS_DIR = Path(__file__).parent.resolve()


def classify_problem(prob_name: str) -> str:
    """根据问题名称归类到对应的 Benchmark 大类。"""
    prob_upper = prob_name.upper()
    if prob_upper.startswith("DC"):
        return "DC-DTLZs"
    elif prob_upper.startswith("C") and "DTLZ" in prob_upper:
        return "C-DTLZs"
    elif prob_upper.startswith("DAS"):
        return "DAS-CMOP"
    elif prob_upper.startswith("LIR"):
        return "LIR-CMOP"
    return "Other-Benchmarks"


def ensure_category_structure(results_dir: Path = RESULTS_DIR):
    """确保每个算法目录下都建立了 4 类 Benchmark 的文件夹，并将未归类的测试问题移入对应分类。"""
    if not results_dir.exists():
        return

    algo_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for algo_dir in algo_dirs:
        # 1. 建立 4 个空分类文件夹
        for cat in BENCHMARK_CATEGORIES.keys():
            cat_dir = algo_dir / cat
            cat_dir.mkdir(exist_ok=True)

        # 2. 如果算法根目录下有如 C1DTLZ1 等问题文件夹，将其整理移入对应分类目录
        for item in list(algo_dir.iterdir()):
            if item.is_dir() and item.name not in BENCHMARK_CATEGORIES.keys() and not item.name.startswith("."):
                cat = classify_problem(item.name)
                target_dir = algo_dir / cat / item.name
                if not target_dir.exists():
                    item.rename(target_dir)
                    print(f"[整理] 移动 {algo_dir.name}/{item.name} -> {algo_dir.name}/{cat}/{item.name}")


def inspect_single_npz(npz_path: Path) -> dict:
    """用【方法一】解析单个 NPZ 文件的内部结构与字段数据。"""
    with np.load(npz_path) as data:
        files = data.files
        info = {
            "file_name": npz_path.name,
            "keys": files,
            "seed": npz_path.stem.split("_")[-1],
            "n_feasible": int(data["n_feasible"]) if "n_feasible" in files else len(data.get("feas_f", [])),
            "igd": float(data["igd"]) if "igd" in files else float("nan"),
            "hv": float(data["hv"]) if "hv" in files else float("nan"),
            "eval_count": int(data["eval_count"]) if "eval_count" in files else 0,
            "elapsed_time": float(data["elapsed_time"]) if "elapsed_time" in files else 0.0,
            "x_shape": data["x"].shape if "x" in files else None,
            "f_shape": data["f"].shape if "f" in files else None,
            "feas_f_shape": data["feas_f"].shape if "feas_f" in files else None,
        }
        return info


def inspect_category(algo_name: str, category: str, results_dir: Path = RESULTS_DIR, detail: bool = False) -> list[dict]:
    """批量查看某算法下某 Benchmark 分类的所有 NPZ 文件。"""
    cat_dir = results_dir / algo_name / category
    records = []

    if not cat_dir.exists():
        print(f"⚠️ 路径不存在: {cat_dir}")
        return records

    # 扫描分类目录下的所有问题文件夹
    prob_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
    
    if not prob_dirs:
        print(f"ℹ️ 算法 [{algo_name}] 在 Benchmark 分类 [{category}] 下暂无运行结果文件。")
        return records

    print("\n" + "=" * 80)
    print(f"🔍 批量查看结果: 算法 [{algo_name}] | Benchmark 分类 [{category}]")
    print("=" * 80)

    for prob_dir in prob_dirs:
        npz_files = sorted(list(prob_dir.glob("*.npz")))
        if not npz_files:
            continue

        igds, hvs, feas_counts, times = [], [], [], []
        
        if detail:
            print(f"\n📌 测试问题: {prob_dir.name} (共 {len(npz_files)} 次独立运行 NPZ 文件)")
            print(f"{'文件名':<20} | {'Seed':<6} | {'IGD':<10} | {'HV':<10} | {'可行解数':<8} | {'耗时(s)':<8} | {'解矩阵 (x)':<12} | {'目标矩阵 (f)':<12}")
            print("-" * 95)

        for npz_file in npz_files:
            info = inspect_single_npz(npz_file)
            igds.append(info["igd"])
            hvs.append(info["hv"])
            feas_counts.append(info["n_feasible"])
            times.append(info["elapsed_time"])

            if detail:
                x_str = str(info["x_shape"]) if info["x_shape"] else "N/A"
                f_str = str(info["f_shape"]) if info["f_shape"] else "N/A"
                print(f"{info['file_name']:<20} | {info['seed']:<6} | {info['igd']:<10.4e} | {info['hv']:<10.4f} | {info['n_feasible']:<8} | {info['elapsed_time']:<8.2f} | {x_str:<12} | {f_str:<12}")

        rec = {
            "Algorithm": algo_name,
            "Category": category,
            "Problem": prob_dir.name,
            "Runs": len(npz_files),
            "IGD_Mean": np.mean(igds),
            "IGD_Std": np.std(igds),
            "HV_Mean": np.mean(hvs),
            "HV_Std": np.std(hvs),
            "Feas_Mean": np.mean(feas_counts),
            "Time_Mean": np.mean(times),
        }
        records.append(rec)

    if records:
        df = pd.DataFrame(records)
        print(f"\n📊 [{algo_name} - {category}] 问题汇总统计摘要:")
        print(df.to_string(index=False, formatters={
            "IGD_Mean": "{:.4e}".format,
            "IGD_Std": "{:.4e}".format,
            "HV_Mean": "{:.4f}".format,
            "HV_Std": "{:.4f}".format,
            "Feas_Mean": "{:.1f}".format,
            "Time_Mean": "{:.2f}s".format,
        }))

    return records


def main():
    parser = argparse.ArgumentParser(description="【方法一】批量查看每个算法在各 Benchmark 分类下的 NPZ 文件数据内容与统计摘要。")
    parser.add_argument("--algo", type=str, default=None, help="指定要查看的算法名称 (如 APSEA, DSOCOL)。若不指定则扫描所有算法。")
    parser.add_argument("--category", type=str, default=None, choices=list(BENCHMARK_CATEGORIES.keys()), help="指定 Benchmark 分类 (C-DTLZs, DC-DTLZs, DAS-CMOP, LIR-CMOP)。")
    parser.add_argument("--detail", action="store_true", help="是否打印单次 seed 运行的详细 NPZ 内容。")
    parser.add_argument("--export-csv", action="store_true", help="是否将统计汇总结果导出为 CSV 文件。")

    args = parser.parse_args()

    # 1. 确保 4 个 Benchmark 分类文件夹已创建并归类
    ensure_category_structure()

    # 2. 确定扫描算法列表
    algo_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    algo_list = [args.algo] if args.algo else [d.name for d in algo_dirs]

    # 3. 确定扫描 Benchmark 分类
    cat_list = [args.category] if args.category else list(BENCHMARK_CATEGORIES.keys())

    all_records = []
    for algo in algo_list:
        for cat in cat_list:
            recs = inspect_category(algo, cat, detail=args.detail)
            all_records.extend(recs)

    # 4. 导出 CSV 逻辑
    if args.export_csv and all_records:
        out_csv = RESULTS_DIR / "npz_batch_inspection_summary.csv"
        pd.DataFrame(all_records).to_csv(out_csv, index=False)
        print(f"\n✅ 批量查看统计汇总已保存至: {out_csv}")


if __name__ == "__main__":
    main()
