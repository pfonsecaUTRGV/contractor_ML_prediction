import pandas as pd
df = pd.read_csv("audits_english.csv")  

print(df[['Kpi', 'wbs', 'contractor', 'Grade']].head())
print(df['Grade'].describe())
print(df.isnull().sum())
