import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import PoissonRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv')

# Drop 'Survey ID'
if 'Survey ID' in df.columns:
    df = df.drop('Survey ID', axis=1)

# Show original shape of dataset
print(f"Original dataset shape: {df.shape}")

# Show original target distribution (including empty cells)
print("Original target distribution:")
target_counts = df['Quality of Life'].value_counts(dropna=False).sort_index()
print(target_counts)

# Fill empty cells with 0 (except target variable)
non_target_cols = [col for col in df.columns if col != 'Quality of Life']
df[non_target_cols] = df[non_target_cols].fillna(0)

# Remove rows with empty target values
df = df.dropna(subset=['Quality of Life'])

# Convert to numeric if not already
df['Quality of Life'] = pd.to_numeric(df['Quality of Life'], errors='coerce')
df = df.dropna(subset=['Quality of Life'])

# Convert to binary classification based on mean
target_mean = df['Quality of Life'].mean()
print(f"Mean value of Quality of Life: {target_mean}")
df['Quality of Life Binary'] = (df['Quality of Life'] >= target_mean).astype(int)

# Drop original target and use binary target
X = df.drop(['Quality of Life', 'Quality of Life Binary'], axis=1)
y = df['Quality of Life Binary']

# Show shape of dataset after processing
print(f"Dataset shape after removing rows with empty target values: {df.shape}")

# Show distribution of binary target after conversion
print("Binary target distribution after conversion:")
print(y.value_counts().sort_index())

# Process features for model compatibility
X_processed = pd.DataFrame()

# Convert all columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        # For string/categorical columns, encode them
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X[col].astype(str))
    else:
        # For numeric columns, keep them as is
        X_processed[col] = X[col]

# Initialize results dictionary
results = {
    'Method': [],
    'CV5 Accuracy': [],
    'Top 5 Features': [],
    'Top 4 Features': []
}

# Random Forest
print("Running Random Forest feature selection...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_processed, y)
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
top_features_rf = [X_processed.columns[i] for i in indices_rf[:5]]
cv5_score_rf = np.mean(cross_val_score(rf, X_processed[top_features_rf], y, cv=5, scoring='accuracy'))

# Remove the highest feature and refit
X_reduced_rf = X_processed.drop(top_features_rf[0], axis=1)
rf_reduced = RandomForestClassifier(random_state=42)
rf_reduced.fit(X_reduced_rf, y)
importances_rf_reduced = rf_reduced.feature_importances_
indices_rf_reduced = np.argsort(importances_rf_reduced)[::-1]
top_features_rf_reduced = [X_reduced_rf.columns[i] for i in indices_rf_reduced[:4]]

results['Method'].append('Random Forest')
results['CV5 Accuracy'].append(cv5_score_rf)
results['Top 5 Features'].append(', '.join(top_features_rf))
results['Top 4 Features'].append(', '.join(top_features_rf_reduced))

# XGBoost
print("Running XGBoost feature selection...")
xgb = XGBClassifier(random_state=42)
xgb.fit(X_processed, y)
importances_xgb = xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1]
top_features_xgb = [X_processed.columns[i] for i in indices_xgb[:5]]
cv5_score_xgb = np.mean(cross_val_score(xgb, X_processed[top_features_xgb], y, cv=5, scoring='accuracy'))

# Remove the highest feature and refit
X_reduced_xgb = X_processed.drop(top_features_xgb[0], axis=1)
xgb_reduced = XGBClassifier(random_state=42)
xgb_reduced.fit(X_reduced_xgb, y)
importances_xgb_reduced = xgb_reduced.feature_importances_
indices_xgb_reduced = np.argsort(importances_xgb_reduced)[::-1]
top_features_xgb_reduced = [X_reduced_xgb.columns[i] for i in indices_xgb_reduced[:4]]

results['Method'].append('XGBoost')
results['CV5 Accuracy'].append(cv5_score_xgb)
results['Top 5 Features'].append(', '.join(top_features_xgb))
results['Top 4 Features'].append(', '.join(top_features_xgb_reduced))

# Feature Agglomeration
print("Running Feature Agglomeration...")
# Feature Agglomeration works only on numerical data
n_clusters = min(5, X_processed.shape[1])
agglo = FeatureAgglomeration(n_clusters=n_clusters)
agglo.fit(X_processed)

# New approach: Calculate variance for each feature and select top features across all clusters
feature_variances = {}
for i, label in enumerate(agglo.labels_):
    feature_name = X_processed.columns[i]
    feature_variance = X_processed[feature_name].var()
    feature_variances[feature_name] = feature_variance

# Sort features by variance across all clusters
sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
top_features_fa = [feature for feature, _ in sorted_features[:5]]

# Evaluate with Random Forest for cross-validation
rf_fa = RandomForestClassifier(random_state=42)
cv5_score_fa = np.mean(cross_val_score(rf_fa, X_processed[top_features_fa], y, cv=5, scoring='accuracy'))

# Remove the highest feature and rerun feature selection on the reduced dataset
X_reduced_fa = X_processed.drop(top_features_fa[0], axis=1)
n_clusters_reduced = min(4, X_reduced_fa.shape[1])
agglo_reduced = FeatureAgglomeration(n_clusters=n_clusters_reduced)
agglo_reduced.fit(X_reduced_fa)

# Calculate variance for each feature in the reduced dataset
feature_variances_reduced = {}
for i, label in enumerate(agglo_reduced.labels_):
    feature_name = X_reduced_fa.columns[i]
    feature_variance = X_reduced_fa[feature_name].var()
    feature_variances_reduced[feature_name] = feature_variance

# Sort features by variance across all clusters in reduced dataset
sorted_features_reduced = sorted(feature_variances_reduced.items(), key=lambda x: x[1], reverse=True)
top_features_fa_reduced = [feature for feature, _ in sorted_features_reduced[:4]]

results['Method'].append('Feature Agglomeration')
results['CV5 Accuracy'].append(cv5_score_fa)
results['Top 5 Features'].append(', '.join(top_features_fa))
results['Top 4 Features'].append(', '.join(top_features_fa_reduced))

# Highly Variable Feature Selection
print("Running Highly Variable Feature Selection...")
# Calculate variance for each feature
variances = X_processed.var().sort_values(ascending=False)
top_features_hvgs = variances.index[:5].tolist()

# Use RandomForest for cross-validation
rf_hvgs = RandomForestClassifier(random_state=42)
cv5_score_hvgs = np.mean(cross_val_score(rf_hvgs, X_processed[top_features_hvgs], y, cv=5, scoring='accuracy'))

# Remove the highest variance feature and refit
X_reduced_hvgs = X_processed.drop(top_features_hvgs[0], axis=1)
variances_reduced = X_reduced_hvgs.var().sort_values(ascending=False)
top_features_hvgs_reduced = variances_reduced.index[:4].tolist()

results['Method'].append('Highly Variable Features')
results['CV5 Accuracy'].append(cv5_score_hvgs)
results['Top 5 Features'].append(', '.join(top_features_hvgs))
results['Top 4 Features'].append(', '.join(top_features_hvgs_reduced))

# Spearman Correlation - PROPERLY FIXED VERSION
print("Running Spearman Correlation feature selection...")
# Calculate correlations for all features
correlations = []
for col in X_processed.columns:
    corr, _ = spearmanr(X_processed[col], y)
    correlations.append((col, abs(corr)))

# Sort by absolute correlation
correlations.sort(key=lambda x: x[1], reverse=True)
top_features_spearman = [feature for feature, corr in correlations[:5]]

# Use RandomForest for cross-validation
rf_spearman = RandomForestClassifier(random_state=42)
cv5_score_spearman = np.mean(cross_val_score(rf_spearman, X_processed[top_features_spearman], y, cv=5, scoring='accuracy'))

# For the reduced set, we take the next 4 highest correlated features (positions 1-4)
# after removing the highest correlated feature (position 0)
top_features_spearman_reduced = [feature for feature, corr in correlations[1:5]]

results['Method'].append('Spearman Correlation')
results['CV5 Accuracy'].append(cv5_score_spearman)
results['Top 5 Features'].append(', '.join(top_features_spearman))
results['Top 4 Features'].append(', '.join(top_features_spearman_reduced))

# Poisson Regression for Feature Selection
print("Running Poisson Regression feature selection...")
# For Poisson regression, we need a non-negative target. Since we have a binary target, we'll use it directly.
# We'll use absolute coefficients to rank feature importance

# Make a copy of the data since Poisson Regression requires non-negative data
X_poisson = X_processed.copy()
# Ensure predictors are positive (add smallest value to make all values non-negative if needed)
for col in X_poisson.columns:
    min_val = X_poisson[col].min()
    if min_val < 0:
        X_poisson[col] = X_poisson[col] - min_val + 0.001  # Small offset to ensure positivity

# Fit Poisson regression
poisson_model = PoissonRegressor(alpha=1.0, max_iter=1000)
poisson_model.fit(X_poisson, y)

# Get absolute coefficients
feature_importances = np.abs(poisson_model.coef_)
indices_poisson = np.argsort(feature_importances)[::-1]
top_features_poisson = [X_processed.columns[i] for i in indices_poisson[:5]]

# Use RandomForest for cross-validation (on original data, not the offset/transformed data)
rf_poisson = RandomForestClassifier(random_state=42)
cv5_score_poisson = np.mean(cross_val_score(rf_poisson, X_processed[top_features_poisson], y, cv=5, scoring='accuracy'))

# Remove the highest feature and refit
X_reduced_poisson = X_processed.drop(top_features_poisson[0], axis=1)
X_reduced_poisson_transformed = X_reduced_poisson.copy()
# Ensure predictors are positive for reduced set
for col in X_reduced_poisson_transformed.columns:
    min_val = X_reduced_poisson_transformed[col].min()
    if min_val < 0:
        X_reduced_poisson_transformed[col] = X_reduced_poisson_transformed[col] - min_val + 0.001

poisson_model_reduced = PoissonRegressor(alpha=1.0, max_iter=1000)
poisson_model_reduced.fit(X_reduced_poisson_transformed, y)

# Get absolute coefficients
feature_importances_reduced = np.abs(poisson_model_reduced.coef_)
indices_poisson_reduced = np.argsort(feature_importances_reduced)[::-1]
top_features_poisson_reduced = [X_reduced_poisson.columns[i] for i in indices_poisson_reduced[:4]]

results['Method'].append('Poisson Regression')
results['CV5 Accuracy'].append(cv5_score_poisson)
results['Top 5 Features'].append(', '.join(top_features_poisson))
results['Top 4 Features'].append(', '.join(top_features_poisson_reduced))

# Create and save results DataFrame
results_df = pd.DataFrame(results)
# Format CV5 Accuracy to 4 effective digits
results_df['CV5 Accuracy'] = results_df['CV5 Accuracy'].map(lambda x: f'{x:.4g}')
results_df.to_csv('result.csv', index=False)

print("Analysis complete. Results saved to result.csv")
