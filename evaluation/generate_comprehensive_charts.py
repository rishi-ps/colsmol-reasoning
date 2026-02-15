"""
Generate comprehensive result charts for the technical report.
Reads all MTEB JSON result files and produces publication-ready PNGs.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
CHARTS = RESULTS / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────────────────────
C_COLSMOL = "#e74c3c"
C_COLPALI = "#3498db"
C_COLQWEN = "#2ecc71"
LANG_COLORS = {"english": "#2c3e50", "french": "#e74c3c", "spanish": "#f39c12", "german": "#8e44ad"}

# ── Load all MTEB JSONs ─────────────────────────────────────────────────────
def load_all():
    """Return dict: task_name -> [{subset metrics dict}, ...]"""
    out = {}
    for f in sorted(glob.glob(str(RESULTS / "**/*.json"), recursive=True)):
        if "model_meta" in f:
            continue
        data = json.load(open(f))
        out[data["task_name"]] = data["scores"]["test"]
    return out

# ── Reference model data (from ColPali / ColQwen2 papers) ───────────────────
V1_TASKS = [
    "ArxivQA", "DocVQA", "InfoVQA", "TabFQuAD", "TATDQA",
    "ShiftProject", "SynDoc-AI", "SynDoc-Energy", "SynDoc-Gov", "SynDoc-Health",
]
V1_JSON_MAP = {
    "VidoreArxivQARetrieval": "ArxivQA",
    "VidoreDocVQARetrieval": "DocVQA",
    "VidoreInfoVQARetrieval": "InfoVQA",
    "VidoreTabfquadRetrieval": "TabFQuAD",
    "VidoreTatdqaRetrieval": "TATDQA",
    "VidoreShiftProjectRetrieval": "ShiftProject",
    "VidoreSyntheticDocQAAIRetrieval": "SynDoc-AI",
    "VidoreSyntheticDocQAEnergyRetrieval": "SynDoc-Energy",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval": "SynDoc-Gov",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval": "SynDoc-Health",
}
REF_COLPALI = {
    "ArxivQA": 79.1, "DocVQA": 54.4, "InfoVQA": 81.8, "TabFQuAD": 83.9,
    "TATDQA": 65.8, "ShiftProject": 73.2, "SynDoc-AI": 96.2,
    "SynDoc-Energy": 91.0, "SynDoc-Gov": 92.7, "SynDoc-Health": 94.4,
}
REF_COLQWEN = {
    "ArxivQA": 86.4, "DocVQA": 56.2, "InfoVQA": 89.8, "TabFQuAD": 88.7,
    "TATDQA": 75.2, "ShiftProject": 85.7, "SynDoc-AI": 98.8,
    "SynDoc-Energy": 94.8, "SynDoc-Gov": 93.6, "SynDoc-Health": 97.3,
}

V2_JSON_MAP = {
    "Vidore2ESGReportsHLRetrieval": "ESG (HL)",
    "Vidore2ESGReportsRetrieval": "ESG Reports",
    "Vidore2EconomicsReportsRetrieval": "Economics",
    "Vidore2BioMedicalLecturesRetrieval": "BioMedical",
}

# ── Chart 1: V1 nDCG@5 grouped bar ─────────────────────────────────────────
def chart_v1_bar(all_data):
    colsmol = {}
    for jname, short in V1_JSON_MAP.items():
        if jname in all_data:
            colsmol[short] = all_data[jname][0]["ndcg_at_5"] * 100

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(V1_TASKS))
    w = 0.25

    cs_vals = [colsmol.get(t, 0) for t in V1_TASKS]
    cp_vals = [REF_COLPALI.get(t, 0) for t in V1_TASKS]
    cq_vals = [REF_COLQWEN.get(t, 0) for t in V1_TASKS]

    ax.bar(x - w, cs_vals, w, label="ColSmol-256M", color=C_COLSMOL, edgecolor="white")
    ax.bar(x, cp_vals, w, label="ColPali-v1.3 (3B)", color=C_COLPALI, edgecolor="white")
    ax.bar(x + w, cq_vals, w, label="ColQwen2 (2B)", color=C_COLQWEN, edgecolor="white")

    for i, v in enumerate(cs_vals):
        ax.text(i - w, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_COLSMOL)
    for i, v in enumerate(cp_vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_COLPALI)
    for i, v in enumerate(cq_vals):
        ax.text(i + w, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_COLQWEN)

    ax.set_xticks(x)
    ax.set_xticklabels(V1_TASKS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("nDCG@5 (%)", fontweight="bold")
    ax.set_ylim(40, 105)
    ax.set_title("ViDoRe v1 — Per-Task nDCG@5", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHARTS / "v1_ndcg5_bar.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v1_ndcg5_bar.png'}")


# ── Chart 2: V1 nDCG heatmap across cutoffs ─────────────────────────────────
def chart_v1_heatmap(all_data):
    cutoffs = [1, 3, 5, 10, 20, 100]
    matrix = []
    for jname in V1_JSON_MAP:
        if jname in all_data:
            s = all_data[jname][0]
            row = [s[f"ndcg_at_{k}"] * 100 for k in cutoffs]
            matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=40, vmax=100)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v < 65 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels([f"@{k}" for k in cutoffs], fontsize=9)
    ax.set_yticks(range(len(V1_TASKS)))
    ax.set_yticklabels(V1_TASKS, fontsize=9)
    ax.set_xlabel("nDCG cutoff", fontweight="bold")
    ax.set_title("ColSmol-256M — ViDoRe v1 nDCG at Various Cutoffs", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="nDCG (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(CHARTS / "v1_ndcg_heatmap.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v1_ndcg_heatmap.png'}")


# ── Chart 3: V1 Recall@k curves ─────────────────────────────────────────────
def chart_v1_recall(all_data):
    ks = [1, 3, 5, 10, 20, 100]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for idx, (jname, short) in enumerate(V1_JSON_MAP.items()):
        if jname not in all_data:
            continue
        s = all_data[jname][0]
        vals = [s[f"recall_at_{k}"] * 100 for k in ks]
        ax.plot(range(len(ks)), vals, "o-", label=short, color=cmap(idx), linewidth=1.5, markersize=4)

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k", fontweight="bold")
    ax.set_ylabel("Recall@k (%)", fontweight="bold")
    ax.set_ylim(35, 105)
    ax.set_title("ColSmol-256M — ViDoRe v1 Recall Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHARTS / "v1_recall_curves.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v1_recall_curves.png'}")


# ── Chart 4: V1 multi-metric radar ──────────────────────────────────────────
def chart_v1_radar(all_data):
    colsmol = {}
    for jname, short in V1_JSON_MAP.items():
        if jname in all_data:
            colsmol[short] = all_data[jname][0]["ndcg_at_5"] * 100

    cats = V1_TASKS
    cs_vals = [colsmol.get(t, 0) for t in cats]
    cp_vals = [REF_COLPALI.get(t, 0) for t in cats]
    cq_vals = [REF_COLQWEN.get(t, 0) for t in cats]

    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for vals, label, color, ls in [
        (cs_vals, "ColSmol-256M", C_COLSMOL, "-"),
        (cp_vals, "ColPali-v1.3 (3B)", C_COLPALI, "--"),
        (cq_vals, "ColQwen2 (2B)", C_COLQWEN, "-."),
    ]:
        v = vals + vals[:1]
        ax.plot(angles, v, ls, linewidth=2, label=label, color=color, marker="o", markersize=4)
        ax.fill(angles, v, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8, fontweight="bold")
    ax.set_ylim(40, 100)
    ax.set_title("ViDoRe v1 — nDCG@5 Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "v1_radar.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v1_radar.png'}")


# ── Chart 5: V1 gap analysis (delta to ColPali) ─────────────────────────────
def chart_v1_gap(all_data):
    colsmol = {}
    for jname, short in V1_JSON_MAP.items():
        if jname in all_data:
            colsmol[short] = all_data[jname][0]["ndcg_at_5"] * 100

    tasks = V1_TASKS
    deltas = [colsmol.get(t, 0) - REF_COLPALI.get(t, 0) for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [C_COLSMOL if d < 0 else C_COLQWEN for d in deltas]
    bars = ax.barh(tasks, deltas, color=colors, edgecolor="white", height=0.6)
    for bar, d in zip(bars, deltas):
        xpos = bar.get_width() + (0.3 if d >= 0 else -0.3)
        ha = "left" if d >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{d:+.1f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ nDCG@5 vs ColPali-v1.3 (3B)", fontweight="bold")
    ax.set_title("ColSmol-256M Gap to ColPali (negative = worse)", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(CHARTS / "v1_gap_to_colpali.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v1_gap_to_colpali.png'}")


# ── Chart 6: V2 English scores bar ──────────────────────────────────────────
def chart_v2_english(all_data):
    tasks_order = ["ESG (HL)", "ESG Reports", "Economics", "BioMedical"]
    scores = {}
    for jname, short in V2_JSON_MAP.items():
        if jname not in all_data:
            continue
        for subset in all_data[jname]:
            lang = subset.get("hf_subset", "default")
            if lang == "english" or lang == "default":
                scores[short] = subset["ndcg_at_5"] * 100
                break

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [scores.get(t, 0) for t in tasks_order]
    bars = ax.bar(tasks_order, vals, color=C_COLSMOL, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("nDCG@5 (%)", fontweight="bold")
    ax.set_title("ColSmol-256M — ViDoRe v2 English Only", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 65)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHARTS / "v2_english_bar.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v2_english_bar.png'}")


# ── Chart 7: V2 per-language breakdown ───────────────────────────────────────
def chart_v2_language(all_data):
    multi_tasks = {
        "Vidore2ESGReportsRetrieval": "ESG Reports",
        "Vidore2EconomicsReportsRetrieval": "Economics",
        "Vidore2BioMedicalLecturesRetrieval": "BioMedical",
    }
    langs = ["english", "french", "spanish", "german"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (jname, short) in zip(axes, multi_tasks.items()):
        if jname not in all_data:
            continue
        lang_scores = {}
        for subset in all_data[jname]:
            lang = subset.get("hf_subset", "default")
            if lang in langs:
                lang_scores[lang] = subset["ndcg_at_5"] * 100

        vals = [lang_scores.get(l, 0) for l in langs]
        colors = [LANG_COLORS[l] for l in langs]
        bars = ax.bar(langs, vals, color=colors, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 60)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("nDCG@5 (%)", fontweight="bold")
    fig.suptitle("ColSmol-256M — ViDoRe v2 Per-Language Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CHARTS / "v2_language_breakdown.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v2_language_breakdown.png'}")


# ── Chart 8: V2 multi-metric table (Recall & MAP alongside nDCG) ────────────
def chart_v2_metric_table(all_data):
    """Horizontal grouped bar: nDCG@5, MAP@5, Recall@5 for V2 English subsets."""
    tasks_order = ["ESG (HL)", "ESG Reports", "Economics", "BioMedical"]
    metrics = {"nDCG@5": "ndcg_at_5", "MAP@5": "map_at_5", "Recall@5": "recall_at_5"}
    metric_colors = {"nDCG@5": "#2c3e50", "MAP@5": "#e67e22", "Recall@5": "#16a085"}

    data_rows = {t: {} for t in tasks_order}
    for jname, short in V2_JSON_MAP.items():
        if jname not in all_data:
            continue
        for subset in all_data[jname]:
            lang = subset.get("hf_subset", "default")
            if lang == "english" or lang == "default":
                for mname, mkey in metrics.items():
                    data_rows[short][mname] = subset[mkey] * 100
                break

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks_order))
    w = 0.22
    for i, (mname, _) in enumerate(metrics.items()):
        vals = [data_rows[t].get(mname, 0) for t in tasks_order]
        bars = ax.bar(x + i * w - w, vals, w, label=mname, color=metric_colors[mname], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks_order, fontsize=10)
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_ylim(0, 65)
    ax.set_title("ColSmol-256M — ViDoRe v2 English: nDCG / MAP / Recall @5", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHARTS / "v2_multi_metric.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved {CHARTS / 'v2_multi_metric.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_data = load_all()
    print(f"Loaded {len(all_data)} task result files\n")

    chart_v1_bar(all_data)
    chart_v1_heatmap(all_data)
    chart_v1_recall(all_data)
    chart_v1_radar(all_data)
    chart_v1_gap(all_data)
    chart_v2_english(all_data)
    chart_v2_language(all_data)
    chart_v2_metric_table(all_data)

    print("\nAll charts generated.")


if __name__ == "__main__":
    main()
