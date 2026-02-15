import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Calculate metrics for each segment
def calculate_segment_metrics(data, segment_col):
    """Calculate accuracy metrics for each segment"""
    results = []
    
    for segment in data[segment_col].unique():
        segment_mask = data[segment_col] == segment
        segment_subset = data[segment_mask]
        
        n_samples = len(segment_subset)
        y_true_seg = segment_subset['y_true']
        y_pred_seg = segment_subset['y_pred']
        
        # Calculate metrics
        acc = accuracy_score(y_true_seg, y_pred_seg)
        
        # Handle cases where only one class is present
        if len(y_true_seg.unique()) > 1 and len(y_pred_seg.unique()) > 1:
            precision = precision_score(y_true_seg, y_pred_seg, zero_division=0)
            recall = recall_score(y_true_seg, y_pred_seg, zero_division=0)
            f1 = f1_score(y_true_seg, y_pred_seg, zero_division=0)
        else:
            precision = recall = f1 = acc  # For single-class cases
        
        # Confusion matrix
        cm = confusion_matrix(y_true_seg, y_pred_seg)
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        
        results.append({
            'segment': segment,
            'n_samples': n_samples,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        })
    
    return pd.DataFrame(results).sort_values('segment')

# Calculate metrics for all segmentation dimensions
activity_metrics = calculate_segment_metrics(segmentation_df, 'activity_level')
activity_metrics['dimension'] = 'Activity Level'

lifetime_metrics = calculate_segment_metrics(segmentation_df, 'lifetime_segment')
lifetime_metrics['dimension'] = 'Lifetime'

engagement_metrics = calculate_segment_metrics(segmentation_df, 'engagement_pattern')
engagement_metrics['dimension'] = 'Engagement Pattern'

early_metrics = calculate_segment_metrics(segmentation_df, 'early_adopter')
early_metrics['dimension'] = 'Early Adopter'

# Combine all metrics
all_segment_metrics = pd.concat([
    activity_metrics, 
    lifetime_metrics, 
    engagement_metrics, 
    early_metrics
], ignore_index=True)

# Display results
print("="*80)
print("MODEL PERFORMANCE BY USER SEGMENT")
print("="*80)

for dim in ['Activity Level', 'Lifetime', 'Engagement Pattern', 'Early Adopter']:
    dim_data = all_segment_metrics[all_segment_metrics['dimension'] == dim]
    print(f"\n{dim.upper()}")
    print("-"*80)
    
    for _, row in dim_data.iterrows():
        print(f"\n{row['segment']}:")
        print(f"  Sample Size: {row['n_samples']}")
        print(f"  Accuracy: {row['accuracy']:.3f}")
        print(f"  Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f} | F1: {row['f1_score']:.3f}")
        print(f"  Confusion Matrix: TN={row['true_negative']}, FP={row['false_positive']}, FN={row['false_negative']}, TP={row['true_positive']}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find segments with best/worst performance
best_accuracy = all_segment_metrics.loc[all_segment_metrics['accuracy'].idxmax()]
worst_accuracy = all_segment_metrics.loc[all_segment_metrics['accuracy'].idxmin()]

print(f"\nBest Performance: {best_accuracy['segment']} ({best_accuracy['dimension']})")
print(f"  Accuracy: {best_accuracy['accuracy']:.3f} | Sample Size: {best_accuracy['n_samples']}")

print(f"\nWorst Performance: {worst_accuracy['segment']} ({worst_accuracy['dimension']})")
print(f"  Accuracy: {worst_accuracy['accuracy']:.3f} | Sample Size: {worst_accuracy['n_samples']}")

# Calculate variance in accuracy across segments
for dim in ['Activity Level', 'Lifetime', 'Engagement Pattern', 'Early Adopter']:
    dim_data = all_segment_metrics[all_segment_metrics['dimension'] == dim]
    acc_variance = dim_data['accuracy'].std()
    print(f"\n{dim} - Accuracy Std Dev: {acc_variance:.3f}")

segment_performance_metrics = all_segment_metrics