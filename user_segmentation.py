import pandas as pd
import numpy as np

# Merge test data with features to access segmentation characteristics
test_indices = X_test.index
segment_data = features.loc[test_indices].copy()

# Add predictions to segment data
segment_data['y_true'] = y_test.values
segment_data['y_pred'] = prediction_model.predict(X_test)

# Create user segments with simpler percentile-based approach
# 1. Activity Level: Low, Medium, High based on total_events
activity_terciles = segment_data['total_events'].quantile([0.33, 0.67])
segment_data['activity_level'] = 'Medium'
segment_data.loc[segment_data['total_events'] <= activity_terciles[0.33], 'activity_level'] = 'Low'
segment_data.loc[segment_data['total_events'] > activity_terciles[0.67], 'activity_level'] = 'High'

# 2. Lifetime: Short, Medium, Long based on lifetime_days
lifetime_terciles = segment_data['lifetime_days'].quantile([0.33, 0.67])
segment_data['lifetime_segment'] = 'Medium'
segment_data.loc[segment_data['lifetime_days'] <= lifetime_terciles[0.33], 'lifetime_segment'] = 'Short'
segment_data.loc[segment_data['lifetime_days'] > lifetime_terciles[0.67], 'lifetime_segment'] = 'Long'

# 3. Engagement Pattern: based on workflow_entropy
entropy_terciles = segment_data['workflow_entropy'].quantile([0.33, 0.67])
segment_data['engagement_pattern'] = 'Balanced'
segment_data.loc[segment_data['workflow_entropy'] <= entropy_terciles[0.33], 'engagement_pattern'] = 'Focused'
segment_data.loc[segment_data['workflow_entropy'] > entropy_terciles[0.67], 'engagement_pattern'] = 'Exploratory'

# 4. Early Adopter: binary split on events_first_7_days
early_median = segment_data['events_first_7_days'].median()
segment_data['early_adopter'] = 'Low Early'
segment_data.loc[segment_data['events_first_7_days'] > early_median, 'early_adopter'] = 'High Early'

print("USER SEGMENTATION COMPLETE")
print("="*60)
print("\nSegment Definitions:")
print(f"\n1. Activity Level (based on total_events):")
print(f"   - Low: <={activity_terciles[0.33]:.1f} events")
print(f"   - Medium: {activity_terciles[0.33]:.1f} - {activity_terciles[0.67]:.1f} events")
print(f"   - High: >{activity_terciles[0.67]:.1f} events")

print(f"\n2. Lifetime (based on lifetime_days):")
print(f"   - Short: <={lifetime_terciles[0.33]:.0f} days")
print(f"   - Medium: {lifetime_terciles[0.33]:.0f} - {lifetime_terciles[0.67]:.0f} days")
print(f"   - Long: >{lifetime_terciles[0.67]:.0f} days")

print(f"\n3. Engagement Pattern (based on workflow_entropy):")
print(f"   - Focused: <={entropy_terciles[0.33]:.2f} (repetitive patterns)")
print(f"   - Balanced: {entropy_terciles[0.33]:.2f} - {entropy_terciles[0.67]:.2f}")
print(f"   - Exploratory: >{entropy_terciles[0.67]:.2f} (diverse exploration)")

print(f"\n4. Early Adopter (based on events_first_7_days):")
print(f"   - Low Early: <={early_median:.0f} events in first 7 days")
print(f"   - High Early: >{early_median:.0f} events in first 7 days")

print("\n" + "="*60)
print("SEGMENT DISTRIBUTION IN TEST SET")
print("="*60)
print(f"\nActivity Level:\n{segment_data['activity_level'].value_counts()}")
print(f"\nLifetime Segment:\n{segment_data['lifetime_segment'].value_counts()}")
print(f"\nEngagement Pattern:\n{segment_data['engagement_pattern'].value_counts()}")
print(f"\nEarly Adopter:\n{segment_data['early_adopter'].value_counts()}")

segmentation_df = segment_data