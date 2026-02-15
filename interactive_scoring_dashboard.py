import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Zerve design system colors
bg_color = '#1D1D20'
text_primary = '#fbfbff'
text_secondary = '#909094'
highlight = '#ffd400'
success_col = '#17b26a'
warning = '#f04438'
colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4']

print("="*80)
print("INTERACTIVE SUCCESS SCORING DASHBOARD")
print("="*80)

# Tier color mapping
tier_colors = {
    'at_risk': warning,
    'developing': '#FFB482',
    'promising': colors[0],
    'successful': success_col
}

# 1. User Distribution Across Tiers
print("\n1. User Distribution Across Success Tiers")
print("-"*80)

tier_counts = user_scores['tier'].value_counts().reindex(tier_order, fill_value=0)
tier_percentages = (tier_counts / len(user_scores) * 100).round(1)

fig1 = plt.figure(figsize=(12, 7), facecolor=bg_color)
ax1 = plt.gca()
ax1.set_facecolor(bg_color)

tier_bar_colors = [tier_colors[tier] for tier in tier_order]
bars1 = ax1.bar(range(len(tier_order)), tier_counts.values, color=tier_bar_colors, alpha=0.9, edgecolor=text_primary, linewidth=1.5)

# Add value labels on bars
for bar, count, pct in zip(bars1, tier_counts.values, tier_percentages.values):
    height = bar.get_height()
    if count > 0:  # Only add label if count is non-zero
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}\n({pct}%)',
                ha='center', va='bottom', color=text_primary, fontsize=12, fontweight='bold')

ax1.set_xticks(range(len(tier_order)))
ax1.set_xticklabels([t.upper().replace('_', ' ') for t in tier_order], fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Users', fontsize=13, color=text_primary, fontweight='bold')
ax1.set_title('User Distribution Across Success Tiers', fontsize=15, color=text_primary, fontweight='bold', pad=20)
ax1.tick_params(colors=text_primary, labelsize=11)
ax1.spines['bottom'].set_color(text_secondary)
ax1.spines['left'].set_color(text_secondary)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

print(f"✓ Tier distribution chart created")
for tier, count, pct in zip(tier_order, tier_counts.values, tier_percentages.values):
    print(f"  {tier.upper():12s}: {int(count):,} users ({pct}%)")

# 2. Success Score Distribution by Tier (only for tiers with data)
print("\n2. Success Score Distribution by Tier")
print("-"*80)

fig2 = plt.figure(figsize=(12, 7), facecolor=bg_color)
ax2 = plt.gca()
ax2.set_facecolor(bg_color)

# Only plot tiers that have data
tiers_with_data = [tier for tier in tier_order if tier in user_scores['tier'].values and len(user_scores[user_scores['tier'] == tier]) > 0]
tier_data = [user_scores[user_scores['tier'] == tier]['success_score'].values for tier in tiers_with_data]

bp = ax2.boxplot(tier_data, labels=[t.upper().replace('_', ' ') for t in tiers_with_data],
                 patch_artist=True, widths=0.6)

# Color each box by tier
for patch, tier in zip(bp['boxes'], tiers_with_data):
    patch.set_facecolor(tier_colors[tier])
    patch.set_alpha(0.7)
    patch.set_edgecolor(text_primary)
    patch.set_linewidth(1.5)

# Style whiskers, caps, medians
for whisker in bp['whiskers']:
    whisker.set(color=text_secondary, linewidth=1.5)
for cap in bp['caps']:
    cap.set(color=text_secondary, linewidth=1.5)
for median in bp['medians']:
    median.set(color=highlight, linewidth=2.5)

ax2.set_ylabel('Success Score (0-100)', fontsize=13, color=text_primary, fontweight='bold')
ax2.set_title('Score Distribution by Success Tier', fontsize=15, color=text_primary, fontweight='bold', pad=20)
ax2.tick_params(colors=text_primary, labelsize=11)
ax2.spines['bottom'].set_color(text_secondary)
ax2.spines['left'].set_color(text_secondary)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.15, color=text_secondary, axis='y')

plt.tight_layout()
plt.show()

print("✓ Score distribution boxplot created")

# 3. Key Metrics Per Tier
print("\n3. Key Metrics Per Tier")
print("-"*80)

# Join with original features to get behavior metrics
tier_metrics = user_scores.merge(ml_features, on='person_id')

tier_behavior = tier_metrics.groupby('tier').agg({
    'total_events': 'mean',
    'events_per_day': 'mean',
    'activity_span_days': 'mean',
    'sessions_first_7days': 'mean',
    'success_score': 'mean'
}).round(2)

# Reindex with only tiers that have data
tier_behavior = tier_behavior.reindex([t for t in tier_order if t in tier_behavior.index])

fig3 = plt.figure(figsize=(14, 7), facecolor=bg_color)
ax3 = plt.gca()
ax3.set_facecolor(bg_color)

tiers_plot = tier_behavior.index.tolist()
x_pos = np.arange(len(tiers_plot))
width = 0.15

# Plot multiple metrics as grouped bars
metrics_to_plot = ['total_events', 'events_per_day', 'activity_span_days', 'sessions_first_7days']
metric_labels = ['Avg Total Events', 'Avg Events/Day', 'Avg Activity Days', 'Avg 7d Sessions']
metric_colors_plot = [colors[0], colors[1], colors[2], colors[4]]

for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, metric_colors_plot)):
    values = tier_behavior[metric].values
    # Normalize for better visualization
    if metric == 'total_events':
        values = values / 10  # Scale down for visualization
        label = label + ' (÷10)'
    
    ax3.bar(x_pos + i*width, values, width, label=label, color=color, alpha=0.85, edgecolor=text_primary, linewidth=0.8)

ax3.set_xticks(x_pos + width * 1.5)
ax3.set_xticklabels([t.upper().replace('_', ' ') for t in tiers_plot], fontsize=11, fontweight='bold')
ax3.set_ylabel('Metric Value', fontsize=13, color=text_primary, fontweight='bold')
ax3.set_title('Key Behavior Metrics by Success Tier', fontsize=15, color=text_primary, fontweight='bold', pad=20)
ax3.legend(fontsize=10, facecolor=bg_color, edgecolor=text_secondary, labelcolor=text_primary, loc='upper left')
ax3.tick_params(colors=text_primary, labelsize=10)
ax3.spines['bottom'].set_color(text_secondary)
ax3.spines['left'].set_color(text_secondary)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.15, color=text_secondary, axis='y')

plt.tight_layout()
plt.show()

print("✓ Tier behavior metrics chart created")
print("\nBehavior Metrics by Tier:")
print(tier_behavior[metrics_to_plot].to_string())

# 4. Confidence Interval Width Analysis
print("\n4. Confidence Interval Analysis")
print("-"*80)

fig4 = plt.figure(figsize=(12, 7), facecolor=bg_color)
ax4 = plt.gca()
ax4.set_facecolor(bg_color)

tier_ci_data = [user_scores[user_scores['tier'] == tier]['ci_width'].values for tier in tiers_with_data]

violin_parts = ax4.violinplot(tier_ci_data, positions=range(len(tiers_with_data)),
                               showmeans=True, showmedians=True, widths=0.7)

# Color violin plots by tier
for pc, tier in zip(violin_parts['bodies'], tiers_with_data):
    pc.set_facecolor(tier_colors[tier])
    pc.set_alpha(0.6)
    pc.set_edgecolor(text_primary)
    pc.set_linewidth(1.5)

ax4.set_xticks(range(len(tiers_with_data)))
ax4.set_xticklabels([t.upper().replace('_', ' ') for t in tiers_with_data], fontsize=11, fontweight='bold')
ax4.set_ylabel('Confidence Interval Width', fontsize=13, color=text_primary, fontweight='bold')
ax4.set_title('Prediction Confidence by Success Tier (95% CI Width)', fontsize=15, color=text_primary, fontweight='bold', pad=20)
ax4.tick_params(colors=text_primary, labelsize=11)
ax4.spines['bottom'].set_color(text_secondary)
ax4.spines['left'].set_color(text_secondary)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.15, color=text_secondary, axis='y')

plt.tight_layout()
plt.show()

print("✓ Confidence interval analysis chart created")

# 5. Score Percentile Distribution
print("\n5. Overall Score Percentile Distribution")
print("-"*80)

fig5 = plt.figure(figsize=(12, 7), facecolor=bg_color)
ax5 = plt.gca()
ax5.set_facecolor(bg_color)

# Create histogram with tier color zones
n_vals, bins, patches = ax5.hist(user_scores['success_score'], bins=40, color=colors[0], 
                                   alpha=0.7, edgecolor=text_primary, linewidth=1)

# Color bars by tier threshold
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge < 25:
        patch.set_facecolor(tier_colors['at_risk'])
    elif left_edge < 50:
        patch.set_facecolor(tier_colors['developing'])
    elif left_edge < 75:
        patch.set_facecolor(tier_colors['promising'])
    else:
        patch.set_facecolor(tier_colors['successful'])

# Add threshold lines
ax5.axvline(x=25, color=text_primary, linestyle='--', linewidth=2, alpha=0.7, label='Tier Thresholds')
ax5.axvline(x=50, color=text_primary, linestyle='--', linewidth=2, alpha=0.7)
ax5.axvline(x=75, color=text_primary, linestyle='--', linewidth=2, alpha=0.7)

ax5.set_xlabel('Success Score (0-100)', fontsize=13, color=text_primary, fontweight='bold')
ax5.set_ylabel('Number of Users', fontsize=13, color=text_primary, fontweight='bold')
ax5.set_title('Success Score Distribution with Tier Boundaries', fontsize=15, color=text_primary, fontweight='bold', pad=20)
ax5.legend(fontsize=10, facecolor=bg_color, edgecolor=text_secondary, labelcolor=text_primary)
ax5.tick_params(colors=text_primary, labelsize=11)
ax5.spines['bottom'].set_color(text_secondary)
ax5.spines['left'].set_color(text_secondary)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

print("✓ Score distribution histogram created")
print(f"  Score percentiles:")
print(f"    25th: {user_scores['success_score'].quantile(0.25):.2f}")
print(f"    50th: {user_scores['success_score'].quantile(0.50):.2f}")
print(f"    75th: {user_scores['success_score'].quantile(0.75):.2f}")
print(f"    90th: {user_scores['success_score'].quantile(0.90):.2f}")

print("\n" + "="*80)
print("✓ Interactive scoring dashboard complete with 5 visualizations")
print("="*80)