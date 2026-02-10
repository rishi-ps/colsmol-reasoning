"""Generate 3 separate PNGs, each with a radar chart + scores table."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def radar_chart(ax, categories, datasets, title, scale_max=100):
    """Draw a radar chart."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    grid_values = np.linspace(0, scale_max, 6)[1:]
    ax.set_yticks(grid_values)
    ax.set_yticklabels([f"{v:.0f}" for v in grid_values], fontsize=6, color="grey")
    ax.set_ylim(0, scale_max)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, fontweight='bold')

    for label, values, color, linestyle in datasets:
        vals = values + values[:1]
        ax.plot(angles, vals, linewidth=2, label=label,
                color=color, linestyle=linestyle, marker='o', markersize=3.5)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8,
              framealpha=0.9, edgecolor='grey')


def scores_table(ax, categories, datasets, fmt=".1f"):
    """Draw a table of scores below the chart."""
    ax.axis('off')

    col_labels = ["Model"] + categories + ["Avg."]
    cell_text = []
    cell_colors = []

    model_colors = {"#e74c3c": "#fde8e8", "#3498db": "#e8f0fe", "#2ecc71": "#e8fce8"}

    for label, values, color, _ in datasets:
        avg = np.mean(values)
        row = [label] + [f"{v:{fmt}}" for v in values] + [f"{avg:{fmt}}"]
        cell_text.append(row)
        bg = model_colors.get(color, "#ffffff")
        cell_colors.append([bg] * len(row))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.5)

    # Make model name column wider
    n_cols = len(col_labels)
    for row_idx in range(len(cell_text) + 1):  # +1 for header
        table[row_idx, 0].set_width(0.22)      # wide model column
        for j in range(1, n_cols):
            table[row_idx, j].set_width(0.78 / (n_cols - 1))

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=7.5)

    # Bold the model name column and average column
    for i in range(1, len(cell_text) + 1):
        table[i, 0].set_text_props(fontweight='bold', fontsize=7.5)
        table[i, len(col_labels) - 1].set_text_props(fontweight='bold')


def make_chart_with_table(filename, categories, datasets, title, scale_max, fmt=".1f"):
    """Create a single PNG with radar chart on top and scores table below."""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.05)

    ax_radar = fig.add_subplot(gs[0], projection='polar')
    ax_table = fig.add_subplot(gs[1])

    radar_chart(ax_radar, categories, datasets, title, scale_max)
    scores_table(ax_table, categories, datasets, fmt)

    # Source footnote
    fig.text(0.5, 0.02,
             "[eval] = from our evaluation  |  [paper] = from ViDoRe v2 paper (Faysse et al.)",
             ha='center', fontsize=7.5, color='#888', style='italic')

    fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def main():
    base = "/home/bs_thesis/colsmol-reasoning"

    # =========================================================================
    # Chart 1: ViDoRe v1
    # =========================================================================
    v1_cats = ["ArxivQ", "DocQ", "InfoQ", "TabF", "TATQ",
               "Shift", "AI", "Energy", "Gov.", "Health."]
    v1_data = [
        ("ColSmol-256M [eval]",
         [72.0, 55.7, 82.5, 62.1, 74.5, 56.2, 94.9, 91.8, 92.6, 95.1],
         "#e74c3c", "-"),
        ("ColPali (Ref. 448) [paper]",
         [79.1, 54.4, 81.8, 83.9, 65.8, 73.2, 96.2, 91.0, 92.7, 94.4],
         "#3498db", "--"),
        ("ColQwen2 (768) [paper]",
         [86.4, 56.2, 89.8, 88.7, 75.2, 85.7, 98.8, 94.8, 93.6, 97.3],
         "#2ecc71", "-."),
    ]
    make_chart_with_table(f"{base}/vidore_v1_chart.png",
                          v1_cats, v1_data, "ViDoRe v1 (nDCG@5)", 100)

    # =========================================================================
    # Chart 2: ViDoRe v2 — All sub-benchmarks
    # =========================================================================
    v2_cats = ["ESG (Manual)", "Insurance", "Ins. Multi.",
               "Economics", "Biomedical", "Bio Multi.",
               "ESG Reports", "ESG Multi.", "Econ. Multi."]
    v2_data = [
        ("ColSmol-256M [paper]",
         [46.0, 50.4, 34.1, 53.4, 53.2, 34.0, 27.2, 31.3, 27.3],
         "#e74c3c", "-"),
        ("ColPali-v1.3 [paper]",
         [51.1, 59.8, 50.1, 51.6, 59.7, 56.5, 57.0, 55.7, 49.9],
         "#3498db", "--"),
        ("ColQwen2.5-v0.2 [paper]",
         [68.4, 60.3, 53.2, 59.8, 63.6, 61.1, 57.4, 57.4, 56.5],
         "#2ecc71", "-."),
    ]
    make_chart_with_table(f"{base}/vidore_v2_all_chart.png",
                          v2_cats, v2_data, "ViDoRe v2 — All (nDCG@5)", 80)

    # =========================================================================
    # Chart 3: ViDoRe v2 — English Only
    # =========================================================================
    v2_en_cats = ["ESG (Manual)", "ESG Reports", "Economics", "BioMedical"]
    v2_en_data = [
        ("ColSmol-256M [eval]",
         [41.8, 45.6, 53.2, 50.4],
         "#e74c3c", "-"),
        ("ColPali-v1.3 [paper]",
         [51.1, 57.0, 51.6, 59.7],
         "#3498db", "--"),
        ("ColQwen2.5-v0.2 [paper]",
         [68.4, 57.4, 59.8, 63.6],
         "#2ecc71", "-."),
    ]
    make_chart_with_table(f"{base}/vidore_v2_english_chart.png",
                          v2_en_cats, v2_en_data,
                          "ViDoRe v2 — English Only (nDCG@5)", 80)

    # =========================================================================
    # Chart 4: ViDoRe v3 — English Only (nDCG@10, from paper Table 9)
    # =========================================================================
    v3_cats = ["C.S.", "Nucl.", "Fin.", "Phar.", "H.R.", "Ind.", "Tele."]
    v3_data = [
        ("ColSmol-256M [paper]",
         [57.4, 36.5, 47.7, 51.4, 46.0, 38.5, 47.5],
         "#e74c3c", "-"),
        ("ColPali [paper]",
         [72.5, 38.1, 43.3, 57.7, 53.3, 47.0, 59.2],
         "#3498db", "--"),
        ("ColQwen2 [paper]",
         [73.5, 44.1, 50.9, 58.1, 54.7, 49.8, 63.2],
         "#2ecc71", "-."),
    ]
    make_chart_with_table(f"{base}/vidore_v3_chart.png",
                          v3_cats, v3_data,
                          "ViDoRe v3 — English Only (nDCG@10)", 80)

    # =========================================================================
    # Combined 2x2 chart: vidore_radar_charts.png
    # =========================================================================
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.5)

    all_charts = [
        (v1_cats, v1_data, "ViDoRe v1 (nDCG@5)", 100),
        (v2_cats, v2_data, "ViDoRe v2 — All (nDCG@5)", 80),
        (v2_en_cats, v2_en_data, "ViDoRe v2 — English Only (nDCG@5)", 80),
        (v3_cats, v3_data, "ViDoRe v3 — English Only (nDCG@10)", 80),
    ]

    for idx, (cats, data, title, smax) in enumerate(all_charts):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col], projection='polar')
        radar_chart(ax, cats, data, title, smax)

    fig.suptitle("Model Comparison on ViDoRe Benchmarks",
                 fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01,
             "[eval] = from our evaluation  |  [paper] = from published papers",
             ha='center', fontsize=8, color='#888', style='italic')

    fig.savefig(f"{base}/vidore_radar_charts.png", dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {base}/vidore_radar_charts.png")


if __name__ == "__main__":
    main()

