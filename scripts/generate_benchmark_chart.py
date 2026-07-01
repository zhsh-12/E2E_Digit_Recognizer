"""Generate benchmark comparison chart for README."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use a clean style - avoid CJK fonts, use English only for chart labels
matplotlib.rcParams['axes.unicode_minus'] = False

base_dir = Path(__file__).resolve().parent.parent
output_dir = base_dir / 'docs' / 'images'
output_dir.mkdir(parents=True, exist_ok=True)

# ========== Data ==========
# Inference latency (ms) - lower is better
models = ['PyTorch FP32', 'ONNX FP32', 'ONNX INT8']
latency_mean = [1.11, 0.85, 0.26]
latency_std  = [0.13, 0.20, 0.04]
colors = ['#ee4c2c', '#ff9800', '#4caf50']

# Model size (KB)
size_labels = ['PyTorch\n.pth', 'ONNX\nFP32', 'ONNX\nINT8']
sizes_kb = [3748.3, 50.6, 361.6]
size_colors = ['#ee4c2c', '#ff9800', '#4caf50']

# ========== Figure 1: Inference Latency ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Latency bar chart ---
x = np.arange(len(models))
bars1 = ax1.bar(x, latency_mean, width=0.5, color=colors, edgecolor='white', linewidth=1.2)
ax1.errorbar(x, latency_mean, yerr=latency_std, fmt='none', capsize=5, color='#333', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars1, latency_mean):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup annotations
ax1.annotate('', xy=(0, latency_mean[0]), xytext=(2, latency_mean[0]),
             arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
ax1.text(1, latency_mean[0] + 0.15, f'4.3x faster', 
         ha='center', fontsize=12, fontweight='bold', color='#e74c3c')

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylabel('Inference Latency (ms) - lower is better', fontsize=11)
ax1.set_title('Single Image Inference Speed\n(CPU, 100 runs avg)', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylim(0, max(latency_mean) * 1.6)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Model size bar chart ---
x2 = np.arange(len(size_labels))
bars2 = ax2.bar(x2, sizes_kb, width=0.5, color=size_colors, edgecolor='white', linewidth=1.2)

# Add value labels
for bar, val in zip(bars2, sizes_kb):
    label = f'{val:.1f} KB' if val < 1000 else f'{val/1024:.1f} MB'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Size reduction annotations
ax2.annotate('', xy=(0, sizes_kb[0]), xytext=(1, sizes_kb[0]),
             arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
ax2.text(0.5, sizes_kb[0] + 300, f'74x smaller', 
         ha='center', fontsize=12, fontweight='bold', color='#e74c3c')

ax2.set_xticks(x2)
ax2.set_xticklabels(size_labels, fontsize=11)
ax2.set_ylabel('Model Size (KB)', fontsize=11)
ax2.set_title('Model Size Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, max(sizes_kb) * 1.3)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2)
output_path = output_dir / 'benchmark_comparison.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"Chart saved to {output_path}")

# ========== Figure 2: Summary table as image ==========
fig2, ax3 = plt.subplots(figsize=(10, 3))
ax3.axis('off')

table_data = [
    ['Metric', 'PyTorch FP32', 'ONNX FP32', 'ONNX INT8 (Quantized)', 'Improvement'],
    ['Inference Latency', '1.11 ms', '0.85 ms', '0.26 ms', '4.3x faster vs PyTorch'],
    ['Model Size', '3,748 KB (3.7 MB)', '50.6 KB', '361.6 KB', '10.4x smaller vs PyTorch'],
    ['Framework', 'PyTorch 2.11', 'ONNX Runtime', 'ONNX Runtime (INT8)', '-'],
    ['Docker Image', '~1.2 GB (with torch)', '~300 MB', '~300 MB', '75% smaller'],
]

table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for j in range(5):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

# Style improvement column
for i in range(1, 4):
    cell = table[i, 4]
    cell.set_facecolor('#e8f5e9')
    cell.set_text_props(fontweight='bold', color='#2e7d32')

output_path2 = output_dir / 'benchmark_table.png'
plt.savefig(str(output_path2), dpi=150, bbox_inches='tight')
print(f"Table saved to {output_path2}")
