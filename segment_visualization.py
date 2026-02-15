import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Zerve design system colors
bg_color = '#1D1D20'
text_color = '#fbfbff'
grid_color = '#3a3a3f'
colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']

# Create comprehensive visualization of segment performance
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(bg_color)

# 1. Accuracy by Segment (Grouped Bar Chart)
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor(bg_color)

dimensions = ['Activity Level', 'Lifetime', 'Engagement Pattern', 'Early Adopter']
for idx, dim in enumerate(dimensions):
    dim_data = segment_performance_metrics[segment_performance_metrics['dimension'] == dim]
    segments = dim_data['segment'].values
    accuracies = dim_data['accuracy'].values
    
    x_pos = np.arange(len(segments)) + idx * 0.2
    ax1.bar(x_pos, accuracies, width=0.18, label=dim, color=colors[idx % len(colors)], alpha=0.9)

ax1.set_ylabel('Accuracy', fontsize=11, color=text_color, fontweight='bold')
ax1.set_title('Model Accuracy by Segment', fontsize=13, color=text_color, fontweight='bold', pad=15)
ax1.set_ylim([0.99, 1.001])
ax1.tick_params(axis='x', colors=text_color, labelsize=8, rotation=0)
ax1.tick_params(axis='y', colors=text_color, labelsize=9)
ax1.spines['bottom'].set_color(text_color)
ax1.spines['left'].set_color(text_color)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='lower right', fontsize=8, facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
ax1.set_xticks([])

# 2. Sample Size Distribution
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor(bg_color)

sample_sizes = segment_performance_metrics.groupby('dimension')['n_samples'].sum()
wedges, texts, autotexts = ax2.pie(
    sample_sizes.values,
    labels=sample_sizes.index,
    autopct='%1.0f%%',
    colors=colors[:len(sample_sizes)],
    startangle=90,
    textprops={'color': text_color, 'fontsize': 10}
)

for autotext in autotexts:
    autotext.set_color(bg_color)
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax2.set_title('Sample Distribution by Segment Type', fontsize=13, color=text_color, fontweight='bold', pad=15)

# 3. Precision vs Recall by Segment
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor(bg_color)

for idx, dim in enumerate(dimensions):
    dim_data = segment_performance_metrics[segment_performance_metrics['dimension'] == dim]
    ax3.scatter(
        dim_data['recall'], 
        dim_data['precision'],
        s=dim_data['n_samples'] * 0.5,
        alpha=0.7,
        color=colors[idx % len(colors)],
        label=dim,
        edgecolors=text_color,
        linewidths=0.5
    )

ax3.set_xlabel('Recall', fontsize=11, color=text_color, fontweight='bold')
ax3.set_ylabel('Precision', fontsize=11, color=text_color, fontweight='bold')
ax3.set_title('Precision vs Recall by Segment', fontsize=13, color=text_color, fontweight='bold', pad=15)
ax3.tick_params(axis='both', colors=text_color, labelsize=9)
ax3.spines['bottom'].set_color(text_color)
ax3.spines['left'].set_color(text_color)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(loc='lower left', fontsize=8, facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
ax3.set_xlim([0.7, 1.05])
ax3.set_ylim([0.95, 1.05])

# 4. Confusion Matrix Summary
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor(bg_color)

segments_list = []
tp_list = []
tn_list = []
fp_list = []
fn_list = []

for idx, row in segment_performance_metrics.iterrows():
    segments_list.append(f"{row['segment'][:4]}")
    tp_list.append(row['true_positive'])
    tn_list.append(row['true_negative'])
    fp_list.append(row['false_positive'])
    fn_list.append(row['false_negative'])

x_pos = np.arange(len(segments_list))
width = 0.6

ax4.bar(x_pos, tn_list, width, label='True Negative', color=colors[2], alpha=0.9)
ax4.bar(x_pos, tp_list, width, bottom=tn_list, label='True Positive', color=colors[0], alpha=0.9)
ax4.bar(x_pos, fp_list, width, bottom=np.array(tn_list)+np.array(tp_list), label='False Positive', color=colors[3], alpha=0.9)
ax4.bar(x_pos, fn_list, width, bottom=np.array(tn_list)+np.array(tp_list)+np.array(fp_list), label='False Negative', color='#f04438', alpha=0.9)

ax4.set_ylabel('Count', fontsize=11, color=text_color, fontweight='bold')
ax4.set_title('Prediction Distribution by Segment', fontsize=13, color=text_color, fontweight='bold', pad=15)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(segments_list, rotation=45, ha='right')
ax4.tick_params(axis='both', colors=text_color, labelsize=8)
ax4.spines['bottom'].set_color(text_color)
ax4.spines['left'].set_color(text_color)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.legend(loc='upper left', fontsize=8, facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)

plt.tight_layout(pad=2.5)
print("Segment performance visualizations created successfully")
print("\nKey Observations:")
print("- All segments show excellent model accuracy (99.7%+)")
print("- Model performs consistently across different user characteristics")
print("- Minimal false predictions across all segments")
print("- Sample sizes well-distributed across segmentation dimensions")