import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("zervedataset1.csv")
print("Dataset loaded successfully")
print(f"Shape: {df.shape}")

# Check available columns
print("\nAvailable columns:")
print(df.columns.tolist())

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

# Identify the correct user identifier column
# Looking at the columns, we have: person_id, distinct_id, prop_$user_id, prop_userId, prop_user_id
# Use person_id as the main user identifier
user_id_col = 'person_id'

# Check if there's an event_type column, otherwise use 'event'
event_col = 'event' if 'event' in df.columns else 'event_type'

print(f"\nUsing '{user_id_col}' as user identifier")
print(f"Using '{event_col}' as event type")

# Sort by user and timestamp
df = df.sort_values([user_id_col, 'timestamp'])

# Create user-level features
user_features = df.groupby(user_id_col).agg(
    total_events=(event_col, 'count'),
    active_days=('timestamp', lambda x: x.dt.date.nunique()),
    first_activity=('timestamp', 'min'),
    last_activity=('timestamp', 'max')
).reset_index()

user_features['lifetime_days'] = (
    user_features['last_activity'] - user_features['first_activity']
).dt.days + 1

# Calculate days from start for each user
df['days_from_start'] = df.groupby(user_id_col)['timestamp'] \
    .transform(lambda x: (x - x.min()).dt.days)

# Extract early engagement features (first 7 days)
early_df = df[df['days_from_start'] <= 7]

early_features = early_df.groupby(user_id_col).agg(
    events_first_7_days=(event_col, 'count'),
    features_used_first_7_days=(event_col, 'nunique')
).reset_index()

# Calculate workflow entropy
from scipy.stats import entropy

def workflow_entropy(actions):
    probs = actions.value_counts(normalize=True)
    return entropy(probs)

workflow_features = df.groupby(user_id_col).agg(
    workflow_entropy=(event_col, workflow_entropy),
    unique_actions=(event_col, 'nunique')
).reset_index()

# Merge all features
features = user_features \
    .merge(early_features, on=user_id_col, how='left') \
    .merge(workflow_features, on=user_id_col, how='left')

features.fillna(0, inplace=True)

# Define success criteria
median_events = features['total_events'].median()
features['successful_user'] = (
    (features['active_days'] >= 20) &
    (features['total_events'] >= median_events)
).astype(int)

print(f"\nTotal users: {len(features)}")
print(f"Successful users: {features['successful_user'].sum()} ({features['successful_user'].mean()*100:.1f}%)")

# Train ML model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = features.drop([user_id_col, 'successful_user', 'first_activity', 'last_activity'], axis=1)
y = features['successful_user']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n" + "="*50)
print("MODEL CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, model.predict(X_test)))

# Feature importance
importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n" + "="*50)
print("TOP 10 FEATURE IMPORTANCES")
print("="*50)
for feat, imp in importances.head(10).items():
    print(f"{feat:35s}: {imp:.4f}")

prediction_model = model