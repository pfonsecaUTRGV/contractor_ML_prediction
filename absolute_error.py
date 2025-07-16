import pandas as pd
import numpy as np

# Load the CSV with predicted and actual values
df = pd.read_csv('predicted_vs_actual_xgboost.csv')  # change file name for each model

# Calculate absolute error
df['AbsoluteError'] = np.abs(df['TrueGrade'] - df['PredictedGrade'])

# Save only the absolute errors to a new CSV
df[['AbsoluteError']].to_csv('abs_xgb_error.csv', index=False, header=False)
