import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

data = json.load(open("results/agent_eval/eval_test_merged.json"))

strategies = [
    "Question", "Self-disclosure", "Restatement\nor Paraphrasing",
    "Information", "Reflection of\nfeelings", "Affirmation and\nReassurance",
    "Others", "Providing\nSuggestions"
]
strategy_keys = [
    "Question", "Self-disclosure", "Restatement or Paraphrasing",
    "Information", "Reflection of feelings", "Affirmation and Reassurance",
    "Others", "Providing Suggestions"
]

per_strategy = data["per_strategy"]
accuracies = [per_strategy[k] for k in strategy_keys]
overall = data["accuracy"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f1117')

# ── 1. Per-strategy bar chart ──────────────────────────────────────────────
ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.38])
ax1.set_facecolor('#1a1d27')

colors = ['#ef4444' if a < 0.1 else '#f97316' if a < 0.2 else '#22c55e' for a in accuracies]
bars = ax1.bar(range(len(strategies)), [a * 100 for a in accuracies],
               color=colors, width=0.6, zorder=3)

ax1.axhline(overall * 100, color='#60a5fa', linewidth=2, linestyle='--',
            label=f'Overall Acc: {overall*100:.1f}%', zorder=4)

for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc*100:.1f}%', ha='center', va='bottom', color='white',
             fontsize=9, fontweight='bold')

ax1.set_xticks(range(len(strategies)))
ax1.set_xticklabels(strategies, color='#cbd5e1', fontsize=8.5)
ax1.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=11)
ax1.set_title('Agent CoT — Per-Strategy Accuracy', color='white', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylim(0, 55)
ax1.tick_params(colors='#94a3b8')
ax1.spines[:].set_color('#334155')
ax1.grid(axis='y', color='#334155', alpha=0.5, zorder=0)
ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=10)

# ── 2. Confusion matrix ────────────────────────────────────────────────────
ax2 = fig.add_axes([0.63, 0.55, 0.35, 0.38])
ax2.set_facecolor('#1a1d27')

confusion = data["confusion"]
short = ["Q", "SD", "R&P", "Info", "RoF", "A&R", "Oth", "PS"]
n = len(strategy_keys)
matrix = np.zeros((n, n))
for i, true_s in enumerate(strategy_keys):
    if true_s in confusion:
        for j, pred_s in enumerate(strategy_keys):
            matrix[i][j] = confusion[true_s].get(pred_s, 0)

# Normalize rows
row_sums = matrix.sum(axis=1, keepdims=True)
matrix_norm = np.divide(matrix, row_sums, where=row_sums != 0)

cmap = LinearSegmentedColormap.from_list('custom', ['#0f1117', '#1e3a5f', '#3b82f6', '#93c5fd'])
im = ax2.imshow(matrix_norm, cmap=cmap, aspect='auto', vmin=0, vmax=0.6)

ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(short, color='#cbd5e1', fontsize=9)
ax2.set_yticklabels(short, color='#cbd5e1', fontsize=9)
ax2.set_xlabel('Predicted', color='#94a3b8', fontsize=10)
ax2.set_ylabel('True', color='#94a3b8', fontsize=10)
ax2.set_title('Confusion Matrix (normalized)', color='white', fontsize=11, fontweight='bold', pad=8)

for i in range(n):
    for j in range(n):
        val = matrix_norm[i][j]
        if val > 0.05:
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     color='white' if val > 0.25 else '#94a3b8', fontsize=7.5)

plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='#94a3b8', labelcolor='#94a3b8')

# ── 3. Summary stats ───────────────────────────────────────────────────────
ax3 = fig.add_axes([0.05, 0.05, 0.9, 0.42])
ax3.set_facecolor('#1a1d27')
ax3.axis('off')

# Compare Agent vs SFT (from memory)
sft_per_strategy = {
    "Question": 0.85, "Self-disclosure": 0.0, "Restatement or Paraphrasing": 0.0,
    "Information": 0.0, "Reflection of feelings": 0.0, "Affirmation and Reassurance": 0.0,
    "Others": 0.0, "Providing Suggestions": 0.0,
}
sft_acc = [sft_per_strategy[k] for k in strategy_keys]
agent_acc = accuracies

x = np.arange(len(strategies))
w = 0.35

ax3.set_facecolor('#1a1d27')
ax3_real = fig.add_axes([0.05, 0.05, 0.9, 0.40])
ax3_real.set_facecolor('#1a1d27')

b1 = ax3_real.bar(x - w/2, [a*100 for a in sft_acc], w, color='#f97316', alpha=0.85, label='SFT v2', zorder=3)
b2 = ax3_real.bar(x + w/2, [a*100 for a in agent_acc], w, color='#22c55e', alpha=0.85, label='Agent CoT', zorder=3)

ax3_real.set_xticks(x)
ax3_real.set_xticklabels(strategies, color='#cbd5e1', fontsize=8.5)
ax3_real.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=11)
ax3_real.set_title('SFT v2 vs Agent CoT — Per-Strategy Comparison', color='white', fontsize=13, fontweight='bold', pad=10)
ax3_real.set_ylim(0, 100)
ax3_real.tick_params(colors='#94a3b8')
ax3_real.spines[:].set_color('#334155')
ax3_real.grid(axis='y', color='#334155', alpha=0.5, zorder=0)
ax3_real.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=11)

# overall labels
ax3_real.text(0.98, 0.96, f'Agent Overall: {overall*100:.1f}%  |  SFT Overall (Q-only bias)',
              transform=ax3_real.transAxes, ha='right', va='top',
              color='#94a3b8', fontsize=9)

# ── Title ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, 'ESC Agent CoT Evaluation — Full Test Set (n=1596)',
         ha='center', color='white', fontsize=15, fontweight='bold')

plt.savefig('results/agent_eval_charts.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
print("图表已保存: results/agent_eval_charts.png")
