from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("C:/Users/gaelm/OneDrive/Desktop/FL_research/Pedro's project/audits_english.csv")

# Clean text: strip, lowercase, etc.
df["wbs"] = df["wbs"].str.strip().str.lower()

# Select features
X = df[["Kpi", "contractor", "wbs"]]
y = df["Grade"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("kpi", OneHotEncoder(handle_unknown="ignore"), ["Kpi"]),
        ("contractor", OneHotEncoder(handle_unknown="ignore"), ["contractor"]),
        ("wbs", TfidfVectorizer(max_features=100, stop_words="english"), "wbs"),
    ]
)

# Pipeline
pipeline_1 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
])

pipeline_2 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
])
# Train pipeline_1
pipeline_1.fit(X_train, y_train)

y_pred=pipeline_1.predict(X_test)

mae= mean_absolute_error(y_test, y_pred)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))

r2=r2_score(y_test,y_pred)

print(f"Mean Absolute Error(Pipeline_1): {mae:.3f}")
print(f"Root Mean Squared Error(Pipeline_1): {rmse:.3f}")
print(f"R^2 Score(Pipeline_1): {r2_score(y_test, y_pred):.3f}")

print(" ")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Grade")
plt.ylabel("Predicted Grade")
plt.title("RandomForest (Pipeline_1): Predicted vs True Grades")
plt.show()

#train pipeline_2
pipeline_2.fit(X_train, y_train)
y_pred_2=pipeline_2.predict(X_test)

mae_2= mean_absolute_error(y_test, y_pred_2)
rmse_2=np.sqrt(mean_squared_error(y_test,y_pred_2))
r2_2=r2_score(y_test,y_pred_2)

print(f"Mean Absolute Error (Pipeline_2):{mae_2:.3f}")
print(f"Root Mean Squared Error (Pipeline_2): {rmse_2:.3f}")
print(f"R^2 Score(Pipeline_2): {r2_score(y_test, y_pred_2):.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_2, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Grade")
plt.ylabel("Predicted Grade")
plt.title("RandomForest(Pipeline_2): Predicted vs True Grades")
plt.show()