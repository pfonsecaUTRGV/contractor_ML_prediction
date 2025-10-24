import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.inspection import permutation_importance

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="No further splits with positive gain")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Loading data set
df = pd.read_csv("audits_english_dates.csv").fillna('')
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['date'])

# Adding time series features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
print(" Added timestamp features.")

# Categorical data encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# Text embeddings 
print("Generating text embeddings for WBS")
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# PCA reduction
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print(f"PCA reduced embeddings to shape: {wbs_reduced.shape}")

# Time features 
time_features = df[['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos']].values
#X = np.hstack([cat_features, wbs_reduced, time_features])
X = np.hstack([cat_features, wbs_reduced])
y = df['Grade'].values

# Ensure target is numeric
if y.dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

# Name features
cat_feature_names = encoder.get_feature_names_out(['Kpi', 'contractor'])
pca_feature_names = [f'WBS_PC{i+1}' for i in range(wbs_reduced.shape[1])]
time_feature_names = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos']
feature_names = np.concatenate([cat_feature_names, pca_feature_names, time_feature_names])
#feature_names = np.concatenate([cat_feature_names, pca_feature_names])



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add standar scaling for models 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DataFrames for LightGBM
X_train_df = pd.DataFrame(X_train, columns=feature_names)  # UNscaled
X_test_df = pd.DataFrame(X_test, columns=feature_names)
X_train_df.columns = X_train_df.columns.str.replace(' ', '_')
X_test_df.columns = X_test_df.columns.str.replace(' ', '_')


# Training models
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
model_lgb = lgb.LGBMRegressor(random_state=42)
model_lgb.fit(X_train_df, y_train)


# Hyperparameter tuning
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1]}
param_grid_lgbm = {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}

# Grid Search
print("\n Tuning Random Forest")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)

print("\n Tuning XGBoost")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGB params:", grid_xgb.best_params_)

print("\n Tuning LightGBM")
grid_lgbm = GridSearchCV(model_lgb, param_grid_lgbm, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_lgbm.fit(X_train_df, y_train)  # Use DataFrame here
best_lgbm = grid_lgbm.best_estimator_
print("Best LightGBM params:", grid_lgbm.best_params_)

# Model testing
y_pred_rf = best_rf.predict(X_test_scaled)
y_pred_xgb = best_xgb.predict(X_test_scaled)
y_pred_lgbm = best_lgbm.predict(X_test_df)

# Printing metrics
def tolerance_accuracy(y_true, y_pred, tolerance=5):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)

results_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'MAE': [
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_xgb),
        mean_absolute_error(y_test, y_pred_lgbm),
    ],
    'RMSE': [
        sqrt(mean_squared_error(y_test, y_pred_rf)),
        sqrt(mean_squared_error(y_test, y_pred_xgb)),
        sqrt(mean_squared_error(y_test, y_pred_lgbm)),
    ],
    'R2': [
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
        r2_score(y_test, y_pred_lgbm),
    ],
    'Accuracy': [
        tolerance_accuracy(y_test, y_pred_rf),
        tolerance_accuracy(y_test, y_pred_xgb),
        tolerance_accuracy(y_test, y_pred_lgbm),
    ]
})

# Print Results
print("\n Final Model Performance:")
display_df = results_df.copy()
display_df[['MAE', 'RMSE', 'R2', 'Accuracy']] = display_df[['MAE', 'RMSE', 'R2', 'Accuracy']].round(3)
print(display_df.to_string(index=False))

# Plots
sns.set(style="whitegrid", font_scale=1.2)
metrics = ['MAE', 'RMSE', 'R2', 'Accuracy']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    sns.barplot(data=results_df, x='Model', y=metric, palette='viridis', ax=ax)
    ax.set_title(f'{metric} by Model')
plt.tight_layout()
plt.show()

# Tolerance Accuracy Curve
tolerance_values = np.arange(0, 11)
plt.figure(figsize=(10, 6))
for model_name, y_pred in zip(results_df['Model'], [y_pred_rf, y_pred_xgb, y_pred_lgbm]):
    accuracies = [tolerance_accuracy(y_test, y_pred, tol) for tol in tolerance_values]
    plt.plot(tolerance_values, accuracies, label=model_name)

plt.xlabel('Tolerance Threshold')
plt.ylabel('Accuracy')
plt.title('Tolerance Accuracy vs Tolerance Threshold')
plt.legend()
plt.grid(True)
plt.show()

'''
# Average Grade Over Time
plt.figure(figsize=(3,5))
df.groupby('year')['Grade'].mean().plot(marker='o')
plt.title('Average Contractor Grade Over Time')
plt.ylabel('Average Grade')
plt.xlabel('Year')
plt.grid(True)
plt.show()
'''