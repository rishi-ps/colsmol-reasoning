"""
Visualize ViDoRe benchmark results using multiple chart types.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read and clean the CSV
df = pd.read_csv('results.csv')
df['Avg.'] = pd.to_numeric(df['Avg.'].astype(str).str.rstrip('.'), errors='coerce')

datasets = ['ArxivQ', 'DocQ', 'InfoQ', 'TabF', 'TATQ', 'Shift', 'AI', 'Energy', 'Gov.', 'Health.']
models = df['Model'].tolist()

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('ViDoRe Benchmark Comparison (nDCG@5)', fontsize=16, fontweight='bold', y=0.98)

# ============ 1. HEATMAP (Best for this data) ============
ax1 = fig.add_subplot(2, 2, 1)
data_matrix = df[datasets].values
im = ax1.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

# Add text annotations
for i in range(len(models)):
    for j in range(len(datasets)):
        val = data_matrix[i, j]
        color = 'white' if val < 70 else 'black'
        ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9, color=color, fontweight='bold')

ax1.set_xticks(np.arange(len(datasets)))
ax1.set_yticks(np.arange(len(models)))
ax1.set_xticklabels(datasets, rotation=45, ha='right')
ax1.set_yticklabels(models)
ax1.set_title('Heatmap View', fontsize=12, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax1, label='nDCG@5 (%)', shrink=0.8)

# ============ 2. RADAR/SPIDER CHART ============
ax2 = fig.add_subplot(2, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
for i, model in enumerate(models):
    values = df[df['Model'] == model][datasets].values.flatten().tolist()
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax2.fill(angles, values, alpha=0.15, color=colors[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(datasets, size=9)
ax2.set_ylim(40, 100)
ax2.set_title('Radar Chart', fontsize=12, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# ============ 3. HORIZONTAL BAR CHART (Clearer than vertical) ============
ax3 = fig.add_subplot(2, 2, 3)
y_pos = np.arange(len(datasets))
bar_height = 0.25

for i, model in enumerate(models):
    scores = df[df['Model'] == model][datasets].values.flatten()
    ax3.barh(y_pos + i * bar_height, scores, bar_height, label=model, color=colors[i], edgecolor='white')

ax3.set_yticks(y_pos + bar_height)
ax3.set_yticklabels(datasets)
ax3.set_xlabel('nDCG@5 (%)', fontweight='bold')
ax3.set_xlim(50, 100)
ax3.set_title('Horizontal Bar Chart', fontsize=12, fontweight='bold', pad=10)
ax3.legend(loc='lower right')
ax3.xaxis.grid(True, linestyle='--', alpha=0.7)

# ============ 4. AVERAGE COMPARISON ============
ax4 = fig.add_subplot(2, 2, 4)
avgs = df['Avg.'].values
bars = ax4.bar(models, avgs, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, avg in zip(bars, avgs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{avg:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax4.set_ylabel('Average nDCG@5 (%)', fontweight='bold')
ax4.set_ylim(70, 92)
ax4.set_title('Overall Average Comparison', fontsize=12, fontweight='bold', pad=10)
ax4.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('results_chart.pdf', bbox_inches='tight', facecolor='white')
print("Saved: results_chart.png and results_chart.pdf")
plt.show()
