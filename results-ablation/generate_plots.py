import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

results_dir = Path(__file__).parent.resolve()
artifact_dir = Path('/Users/jingzeng/.gemini/antigravity/brain/e26117dc-3a6f-4c08-910a-e4d6d0dc698d')
artifact_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(results_dir / 'per_problem_metrics_detail.csv')

# Correct problem objective counts
problems_info = [
    {"name": "C1DTLZ1", "category": "C-DTLZs", "note": "(3-Obj)", "cat_idx": 0},
    {"name": "DC1DTLZ1", "category": "DC-DTLZs", "note": "(3-Obj)", "cat_idx": 1},
    {"name": "DASCMOP1", "category": "DAS-CMOP", "note": "(2-Obj)", "cat_idx": 2},
    {"name": "DASCMOP7", "category": "DAS-CMOP", "note": "(3-Obj)", "cat_idx": 2}, # 3-Obj
    {"name": "LIRCMOP1", "category": "LIR-CMOP", "note": "(2-Obj)", "cat_idx": 3},
    {"name": "LIRCMOP13", "category": "LIR-CMOP", "note": "(3-Obj)", "cat_idx": 3}, # 3-Obj
]

algos = ["DSOCOL", "DSOCOL1", "DSOCOL3", "DSOCOL4"]
algo_labels = {
    "DSOCOL": "DSOCOL (Full)",
    "DSOCOL1": "w/o NGSS",
    "DSOCOL3": "w/o COL",
    "DSOCOL4": "w/o Trend"
}

colors = {
    "DSOCOL": "#2563EB",   # Vibrant Blue (Full Model)
    "DSOCOL1": "#F59E0B",  # Amber/Orange (w/o NGSS)
    "DSOCOL3": "#10B981",  # Emerald/Green (w/o COL)
    "DSOCOL4": "#8B5CF6"   # Purple/Indigo (w/o Trend)
}

edge_colors = {
    "DSOCOL": "#1E40AF",
    "DSOCOL1": "#D97706",
    "DSOCOL3": "#059669",
    "DSOCOL4": "#7C3AED"
}

hatches = {
    "DSOCOL": "///",
    "DSOCOL1": "..",
    "DSOCOL3": "\\\\\\",
    "DSOCOL4": "xx"
}

def format_val(val, metric="IGD"):
    if np.isnan(val) or val == 0:
        return "0.000" if metric == "HV" else "0.0000"
    if val < 0.01:
        return f"{val:.4f}"
    elif val < 0.1:
        return f"{val:.4f}"
    elif val < 1.0:
        return f"{val:.3f}"
    else:
        return f"{val:.3f}"

def plot_deepseek_percentage_chart(metric_type="IGD", output_filename="igd_deepseek_style.png"):
    fig, ax = plt.subplots(figsize=(14.5, 7.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    n_problems = len(problems_info)
    n_algos = len(algos)
    
    bar_width = 0.18
    gap_between_groups = 0.45
    
    x_positions = []
    current_x = 0
    for i in range(n_problems):
        x_positions.append(current_x)
        current_x += (n_algos * bar_width) + gap_between_groups
        
    x_positions = np.array(x_positions)

    mean_col = f"{metric_type}_Mean"

    raw_data = np.zeros((n_problems, n_algos))
    scores = np.zeros((n_problems, n_algos))

    for p_idx, p_info in enumerate(problems_info):
        p_name = p_info["name"]
        for a_idx, algo in enumerate(algos):
            row = df[(df["Problem"] == p_name) & (df["Algorithm"] == algo)]
            if not row.empty:
                raw_data[p_idx, a_idx] = row[mean_col].values[0]

        # Calculate percentage score (100% = Best)
        if metric_type == "IGD":
            best_val = np.min(raw_data[p_idx, :])
            for a_idx in range(n_algos):
                v = raw_data[p_idx, a_idx]
                scores[p_idx, a_idx] = (best_val / v * 100.0) if v > 0 else 0.0
        else: # HV
            max_val = np.max(raw_data[p_idx, :])
            for a_idx in range(n_algos):
                v = raw_data[p_idx, a_idx]
                scores[p_idx, a_idx] = (v / max_val * 100.0) if max_val > 0 else 0.0

    # Draw bars using Percentage Scores (%)
    for a_idx, algo in enumerate(algos):
        offset = (a_idx - (n_algos - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset,
            scores[:, a_idx],
            width=bar_width * 0.9,
            color=colors[algo],
            edgecolor=edge_colors[algo],
            hatch=hatches[algo],
            linewidth=1.2,
            zorder=3,
            label=algo_labels[algo]
        )

        for p_idx, bar in enumerate(bars):
            raw_val = raw_data[p_idx, a_idx]
            sc_val = scores[p_idx, a_idx]
            raw_str = format_val(raw_val, metric_type)
            h = bar.get_height()
            
            if metric_type == "IGD":
                is_best = (raw_val == np.min(raw_data[p_idx, :]))
            else:
                is_best = (raw_val == np.max(raw_data[p_idx, :]))

            fontweight = 'bold' if (algo == "DSOCOL" or is_best) else 'normal'
            textcolor = '#1E3A8A' if algo == "DSOCOL" else ('#0F172A' if is_best else '#475569')

            # Elegant 2-line annotation: Line 1 = Raw Value, Line 2 = Percentage
            label_text = f"{raw_str}\n({sc_val:.1f}%)" if sc_val > 0 else "0.000\n(0.0%)"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 2.0,
                label_text,
                ha='center',
                va='bottom',
                fontsize=8.0,
                fontweight=fontweight,
                color=textcolor,
                zorder=4
            )

    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    if metric_type == "IGD":
        ylabel_text = "Relative IGD Score (%) [100% = Best Performance]\nFormula: (Min_IGD / IGD) × 100%"
    else:
        ylabel_text = "Relative HV Score (%) [100% = Best Performance]\nFormula: (HV / Max_HV) × 100%"
        
    ax.set_ylabel(ylabel_text, fontsize=10.5, fontweight='bold', color='#334155', labelpad=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#94A3B8')
    ax.spines['bottom'].set_linewidth(1.0)

    ax.set_xticks(x_positions)
    xtick_labels = [f"{p['name']}\n{p['note']}" for p in problems_info]
    ax.set_xticklabels(xtick_labels, fontsize=10.5, fontweight='bold', color='#0F172A')

    # Category dividers
    cat_boundaries = [0.5, 1.5, 3.5]
    for b_idx in cat_boundaries:
        pos = (x_positions[int(b_idx)] + x_positions[int(b_idx) + 1]) / 2.0
        ax.axvline(x=pos, color='#CBD5E1', linestyle='--', linewidth=1.2, zorder=2)

    cat_ranges = [
        ("C-DTLZs Benchmark", 0, 0),
        ("DC-DTLZs Benchmark", 1, 1),
        ("DAS-CMOP Benchmark Series", 2, 3),
        ("LIR-CMOP Benchmark Series", 4, 5),
    ]

    bracket_y = -22
    bracket_text_y = -32

    for cat_name, start_p, end_p in cat_ranges:
        x_start = x_positions[start_p] - bar_width * 2.2
        x_end = x_positions[end_p] + bar_width * 2.2
        
        ax.plot([x_start, x_start, x_end, x_end], 
                [bracket_y + 2.5, bracket_y, bracket_y, bracket_y + 2.5], 
                color='#475569', linewidth=1.2, clip_on=False)
        
        ax.text((x_start + x_end) / 2.0, bracket_text_y, cat_name, 
                ha='center', va='top', fontsize=10.5, fontweight='bold', color='#1E293B', clip_on=False)

    handles = []
    for algo in algos:
        patch = mpatches.Patch(
            facecolor=colors[algo],
            edgecolor=edge_colors[algo],
            hatch=hatches[algo],
            label=algo_labels[algo]
        )
        handles.append(patch)

    legend = ax.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        handlelength=2.5,
        handleheight=1.2
    )

    metric_desc = "IGD (Lower Raw Value is Better ↓)" if metric_type == "IGD" else "HV (Higher Raw Value is Better ↑)"
    fig.suptitle(f"Algorithm Benchmark Relative Performance Comparison: {metric_desc}",
                 fontsize=13.5, fontweight='bold', color='#0F172A', y=0.98)

    plt.tight_layout(rect=[0, 0.09, 1, 0.95])
    
    out_path = results_dir / output_filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    try:
        shutil.copy(out_path, artifact_dir / output_filename)
    except Exception:
        pass
    plt.close(fig)
    print(f"Saved percentage DeepSeek style chart: {out_path}")

def plot_multi_panel_raw(metric_type="IGD", output_filename="igd_subplot_grid.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), dpi=300)
    axes = axes.flatten()
    fig.patch.set_facecolor('#FFFFFF')

    mean_col = f"{metric_type}_Mean"
    std_col = f"{metric_type}_Std"

    bar_width = 0.55
    x_indices = np.arange(len(algos))

    for p_idx, p_info in enumerate(problems_info):
        ax = axes[p_idx]
        ax.set_facecolor('#FFFFFF')
        p_name = p_info["name"]

        vals = []
        errs = []
        for algo in algos:
            row = df[(df["Problem"] == p_name) & (df["Algorithm"] == algo)]
            val = row[mean_col].values[0] if not row.empty else 0.0
            err = row[std_col].values[0] if not row.empty else 0.0
            vals.append(val)
            errs.append(err)

        bars = ax.bar(
            x_indices,
            vals,
            yerr=errs,
            capsize=4,
            width=bar_width,
            color=[colors[a] for a in algos],
            edgecolor=[edge_colors[a] for a in algos],
            hatch=[hatches[a] for a in algos],
            linewidth=1.1,
            zorder=3,
            error_kw={'ecolor': '#334155', 'linewidth': 1.2}
        )

        for a_idx, bar in enumerate(bars):
            v = vals[a_idx]
            val_str = format_val(v, metric_type)
            h = bar.get_height()
            
            if metric_type == "IGD":
                is_best = (v == np.min(vals))
            else:
                is_best = (v == np.max(vals))

            fontweight = 'bold' if (algos[a_idx] == "DSOCOL" or is_best) else 'normal'
            textcolor = '#1E3A8A' if algos[a_idx] == "DSOCOL" else ('#0F172A' if is_best else '#475569')

            y_pos = h + errs[a_idx] + (max(vals) * 0.03 if max(vals) > 0 else 0.01)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                val_str,
                ha='center',
                va='bottom',
                fontsize=8.5,
                fontweight=fontweight,
                color=textcolor,
                zorder=4
            )

        ax.set_title(f"{p_name} {p_info['note']} ({p_info['category']})", fontsize=10.5, fontweight='bold', color='#0F172A', pad=8)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([algo_labels[a] for a in algos], fontsize=8.5, rotation=15)
        ax.grid(axis='y', linestyle='--', linewidth=0.7, color='#E2E8F0', zorder=1)
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94A3B8')
        ax.spines['bottom'].set_color('#94A3B8')

        max_y = (max(vals) + max(errs)) if (max(vals) + max(errs)) > 0 else 1.0
        ax.set_ylim(0, max_y * 1.3)

    title_suffix = "(Raw Mean ± Std, Lower is Better ↓)" if metric_type == "IGD" else "(Raw Mean ± Std, Higher is Better ↑)"
    fig.suptitle(f"Detailed Raw Metric Analysis per Benchmark: {metric_type} {title_suffix}",
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    out_path = results_dir / output_filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    try:
        shutil.copy(out_path, artifact_dir / output_filename)
    except Exception:
        pass
    plt.close(fig)
    print(f"Saved raw multi-panel chart: {out_path}")

# Run chart generation with Percentage Scores for DeepSeek charts
plot_deepseek_percentage_chart("IGD", "igd_deepseek_style.png")
plot_deepseek_percentage_chart("HV", "hv_deepseek_style.png")
plot_multi_panel_raw("IGD", "igd_subplot_grid.png")
plot_multi_panel_raw("HV", "hv_subplot_grid.png")
