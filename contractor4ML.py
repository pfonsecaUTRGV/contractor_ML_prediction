import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import xgboost as xgb
import sklearn
import matplotlib.pyplot as plt

# === Load and preprocess data ===
df = pd.read_csv("audits_english.csv")
df = df.fillna('')

# OneHotEncoder (version check for sparse parameter)
version = sklearn.__version__
major, minor, *_ = version.split(".")
minor = int(minor)
if minor >= 2:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# BERT embeddings for WBS
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# PCA to reduce dimensionality of embeddings
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)

# Combine categorical + embeddings features
X_cat = cat_features
X_emb = wbs_reduced
X = np.hstack([X_cat, X_emb])
y = df['Grade'].values

# === Split data ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42
)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === Define hyperparameter search space ===
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# === Initialize XGB regressor ===
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    eval_metric='rmse'
)

# === Hyperparameter tuning WITHOUT early stopping ===
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=25,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    refit=True,
    error_score='raise'
)

print("Starting hyperparameter tuning without early stopping...")
random_search.fit(X_train_scaled, y_train)

print("Best params found:", random_search.best_params_)

# === Train final model WITH early stopping ===

best_params = random_search.best_params_

final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    eval_metric='rmse',
    **best_params
)

# Combine train + val for final training
X_trainval_scaled = scaler.transform(np.vstack([X_train, X_val]))
y_trainval_combined = np.concatenate([y_train, y_val])

print("\nTraining final model with early stopping on train+val set...")
dtrain = xgb.DMatrix(X_trainval_scaled, label=y_trainval_combined)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

final_model_core = xgb.train(
    params={
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        **best_params
    },
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=20,
    verbose_eval=True
)

# Predict using the trained booster
y_pred_test = final_model_core.predict(dtest)


# === Evaluate on test set ===
#y_pred_test = final_model.predict(X_test_scaled)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"\nFinal XGBoost model test results:")
print(f"MAE: {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R2: {r2_test:.2f}")

# === Plot predictions vs true ===
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Grade")
plt.ylabel("Predicted Grade")
plt.title("XGBoost Tuned + Early Stopping: Predicted vs True Grades")
plt.show()
