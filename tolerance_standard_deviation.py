import pandas as pd
import numpy as np

# Define function
def tolerance_accuracy(y_true, y_pred, tolerance):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)

# Define models and their CSV filenames
model_files = {
    "Random Forest": "predicted_vs_actual_randomforest.csv",
    "XGBoost": "predicted_vs_actual_xgboost.csv",
    "SVR": "predicted_vs_actual_svr.csv",
    "LightGBM": "predicted_vs_actual_lightgbm.csv"
}

# Tolerances to check
tolerances = [0,1, 3, 5,7,9]

# Collect results
results = []

for model_name, filename in model_files.items():
    df = pd.read_csv(filename)
    y_true = df['TrueGrade'].values
    y_pred = df['PredictedGrade'].values
    
    accs = [tolerance_accuracy(y_true, y_pred, t) for t in tolerances]
    std_dev = np.std(accs)
    
    results.append([model_name] + accs + [std_dev])

# Create DataFrame
df_results = pd.DataFrame(
    results,
    columns=["Model","Acc@0", "Acc@1", "Acc@3", "Acc@5","Acc@7","Acc@10", "Std Dev"]
)

# Print result
print(df_results)

# Export to LaTeX
latex_code = df_results.to_latex(
    index=False,
    float_format="%.3f",
    caption="Tolerance Accuracy at thresholds 1, 3, and 5 with standard deviation.",
    label="tab:tolerance_std"
)

with open("tolerance_accuracy_table.tex", "w") as f:
    f.write(latex_code)
