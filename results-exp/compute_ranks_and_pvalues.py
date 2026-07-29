"""compute_ranks_and_pvalues.py

Calculates IGD Ranking, HV Ranking, Wilcoxon rank-sum test p-values (+/=/--),
and Friedman ANOVA test statistics for 4 constrained multi-objective algorithms across 6 benchmark problems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ranksums, friedmanchisquare

RESULTS_DIR = Path(__file__).parent.resolve()
ALGOS = ["DSOCOL", "APSEA", "CMOCSO", "IMCMOEAD"]
ALGO_LABELS = {
    "DSOCOL": "DSOCOL (Ours)",
    "APSEA": "APSEA",
    "CMOCSO": "CMOCSO",
    "IMCMOEAD": "IM-C-MOEA/D"
}

PROBLEMS_INFO = [
    {"name": "C1DTLZ1", "category": "C-DTLZs", "objs": 3},
    {"name": "DC1DTLZ1", "category": "DC-DTLZs", "objs": 3},
    {"name": "DASCMOP1", "category": "DAS-CMOP", "objs": 2},
    {"name": "DASCMOP7", "category": "DAS-CMOP", "objs": 3},
    {"name": "LIRCMOP1", "category": "LIR-CMOP", "objs": 2},
    {"name": "LIRCMOP13", "category": "LIR-CMOP", "objs": 3},
]

CONTROL_ALGO = "DSOCOL"


def load_raw_data() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        algo_path = RESULTS_DIR / algo
        for npz_file in algo_path.glob("**/*.npz"):
            prob_name = npz_file.parent.name
            seed = npz_file.stem.split("_")[-1]
            with np.load(npz_file) as data:
                n_feas = int(data["n_feasible"]) if "n_feasible" in data else 0
                igd = float(data["igd"]) if ("igd" in data and not np.isnan(data["igd"])) else np.nan
                hv = float(data["hv"]) if ("hv" in data and not np.isnan(data["hv"])) else np.nan
                
                rows.append({
                    "Algorithm": algo,
                    "Problem": prob_name,
                    "Seed": seed,
                    "N_Feasible": n_feas,
                    "IGD": igd,
                    "HV": hv
                })
    return pd.DataFrame(rows)


def analyze_data():
    df_raw = load_raw_data()
    problems = [p["name"] for p in PROBLEMS_INFO]

    # Containers for summary tables
    detailed_records = []
    
    # Ranks per problem
    igd_ranks = {a: [] for a in ALGOS}
    hv_ranks = {a: [] for a in ALGOS}
    
    # Wilcoxon summary counters vs DSOCOL: (win, tie, loss)
    # win (+) : DSOCOL is significantly better than competitor
    # tie (=) : no significant difference
    # loss (-): DSOCOL is significantly worse than competitor
    wilcoxon_counts_igd = {a: {"+": 0, "=": 0, "-": 0} for a in ALGOS if a != CONTROL_ALGO}
    wilcoxon_counts_hv = {a: {"+": 0, "=": 0, "-": 0} for a in ALGOS if a != CONTROL_ALGO}

    # Data matrix for Friedman test (problems x algorithms)
    friedman_matrix_igd = {a: [] for a in ALGOS}
    friedman_matrix_hv = {a: [] for a in ALGOS}

    for p_info in PROBLEMS_INFO:
        p_name = p_info["name"]
        cat = p_info["category"]
        p_df = df_raw[df_raw["Problem"] == p_name]

        # Calculate Mean and Std for each algorithm
        igd_means = {}
        igd_stds = {}
        hv_means = {}
        hv_stds = {}

        for a in ALGOS:
            sub = p_df[p_df["Algorithm"] == a]
            v_igd = sub["IGD"].dropna().values
            v_hv = sub["HV"].dropna().values

            # If no feasible solutions found (or all NaN)
            igd_m = float(np.mean(v_igd)) if len(v_igd) > 0 else 9999.0
            igd_s = float(np.std(v_igd)) if len(v_igd) > 0 else 0.0
            hv_m = float(np.mean(v_hv)) if len(v_hv) > 0 else 0.0
            hv_s = float(np.std(v_hv)) if len(v_hv) > 0 else 0.0

            igd_means[a] = igd_m
            igd_stds[a] = igd_s
            hv_means[a] = hv_m
            hv_stds[a] = hv_s

            friedman_matrix_igd[a].append(igd_m)
            friedman_matrix_hv[a].append(hv_m)

        # Calculate per-problem ranks (1 is best)
        sorted_igd = sorted(ALGOS, key=lambda x: igd_means[x])
        sorted_hv = sorted(ALGOS, key=lambda x: hv_means[x], reverse=True)

        for a in ALGOS:
            r_igd = sorted_igd.index(a) + 1
            r_hv = sorted_hv.index(a) + 1
            igd_ranks[a].append(r_igd)
            hv_ranks[a].append(r_hv)

        # Perform Wilcoxon rank-sum test vs DSOCOL
        c_sub = p_df[p_df["Algorithm"] == CONTROL_ALGO]
        c_igd = c_sub["IGD"].fillna(9999.0).values
        c_hv = c_sub["HV"].fillna(0.0).values

        for a in ALGOS:
            a_sub = p_df[p_df["Algorithm"] == a]
            a_igd = a_sub["IGD"].fillna(9999.0).values
            a_hv = a_sub["HV"].fillna(0.0).values

            r_igd = sorted_igd.index(a) + 1
            r_hv = sorted_hv.index(a) + 1

            if a == CONTROL_ALGO:
                p_val_igd, sym_igd = 1.0, "="
                p_val_hv, sym_hv = 1.0, "="
            else:
                # Wilcoxon rank-sum for IGD (smaller is better)
                if np.array_equal(c_igd, a_igd):
                    p_val_igd, sym_igd = 1.0, "="
                else:
                    _, p_val_igd = ranksums(c_igd, a_igd)
                    if p_val_igd < 0.05:
                        sym_igd = "+" if igd_means[CONTROL_ALGO] < igd_means[a] else "-"
                    else:
                        sym_igd = "="
                wilcoxon_counts_igd[a][sym_igd] += 1

                # Wilcoxon rank-sum for HV (larger is better)
                if np.array_equal(c_hv, a_hv):
                    p_val_hv, sym_hv = 1.0, "="
                else:
                    _, p_val_hv = ranksums(c_hv, a_hv)
                    if p_val_hv < 0.05:
                        sym_hv = "+" if hv_means[CONTROL_ALGO] > hv_means[a] else "-"
                    else:
                        sym_hv = "="
                wilcoxon_counts_hv[a][sym_hv] += 1

            detailed_records.append({
                "Category": cat,
                "Problem": p_name,
                "Algorithm": a,
                "IGD_Mean": igd_means[a],
                "IGD_Std": igd_stds[a],
                "IGD_Rank": r_igd,
                "IGD_PValue": p_val_igd,
                "IGD_Symbol": sym_igd,
                "HV_Mean": hv_means[a],
                "HV_Std": hv_stds[a],
                "HV_Rank": r_hv,
                "HV_PValue": p_val_hv,
                "HV_Symbol": sym_hv,
            })

    df_detailed = pd.DataFrame(detailed_records)

    # Calculate Overall Average Rank across problems
    avg_igd_ranks = {a: float(np.mean(igd_ranks[a])) for a in ALGOS}
    avg_hv_ranks = {a: float(np.mean(hv_ranks[a])) for a in ALGOS}

    # Perform Friedman test across 6 benchmark problems
    f_stat_igd, f_pval_igd = friedmanchisquare(*[friedman_matrix_igd[a] for a in ALGOS])
    f_stat_hv, f_pval_hv = friedmanchisquare(*[friedman_matrix_hv[a] for a in ALGOS])

    # Build Summary Ranking DataFrame
    summary_rows = []
    for a in ALGOS:
        w_igd = f"{wilcoxon_counts_igd[a]['+']}/{wilcoxon_counts_igd[a]['=']}/{wilcoxon_counts_igd[a]['-']}" if a != CONTROL_ALGO else "N/A"
        w_hv = f"{wilcoxon_counts_hv[a]['+']}/{wilcoxon_counts_hv[a]['=']}/{wilcoxon_counts_hv[a]['-']}" if a != CONTROL_ALGO else "N/A"

        summary_rows.append({
            "Algorithm": a,
            "Label": ALGO_LABELS[a],
            "Avg_IGD_Rank": avg_igd_ranks[a],
            "IGD_Wilcoxon_W/T/L": w_igd,
            "Avg_HV_Rank": avg_hv_ranks[a],
            "HV_Wilcoxon_W/T/L": w_hv,
        })
    df_summary = pd.DataFrame(summary_rows)

    # Save to CSV
    df_detailed.to_csv(RESULTS_DIR / "per_problem_metrics_detail.csv", index=False)
    df_summary.to_csv(RESULTS_DIR / "ranking_and_pvalues_summary.csv", index=False)

    print("=== PER-PROBLEM DETAILED METRICS & RANKS ===")
    print(df_detailed[["Problem", "Algorithm", "IGD_Mean", "IGD_Rank", "IGD_Symbol", "HV_Mean", "HV_Rank", "HV_Symbol"]].to_string())

    print("\n=== OVERALL RANKING & STATISTICAL SUMMARY ===")
    print(df_summary.to_string(index=False))

    print(f"\nFriedman Test IGD: Stat = {f_stat_igd:.4f}, p-value = {f_pval_igd:.4e}")
    print(f"Friedman Test HV:  Stat = {f_stat_hv:.4f}, p-value = {f_pval_hv:.4e}")

    return df_detailed, df_summary, (f_stat_igd, f_pval_igd), (f_stat_hv, f_pval_hv)


if __name__ == "__main__":
    analyze_data()
