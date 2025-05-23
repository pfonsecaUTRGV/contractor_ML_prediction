# Contractor ML Prediction 

This projects aims to use data gather from a construction project for Machine Learning Applications. This data contains information regarding the performance of several contractors. 

## Data organization
The Data contains 4 relevant columns:
-Contractor: Contractor name
-Kpw: Refers to the activity performed
-WBS: A basic description of the general project
-Grade: The grade that the contractor got for that specific activity (KPI)

## Exsisting files
-audits_english.csv: The data set file in comma separated value format
-contractor3ML.py: Script for training, evaluating and ploting Linear Regression, Random Forest, XGboost
-contractor4ML.py: Script for training, evaluating and ploting XGboost. This scriot contains fine tunning and a stopper if the accuracy starts to deprecate
-data_debug.py: Due to the fact that the data set contains categorical data (strings) this needs to be encoded for use in machine learning (ML models can only use numeric data). This file test the performance of the encoding process 
-data_sanity_check.py: Script to check the usabily of the data. Range check, mismachtches, missing values, cardinality, etc.

## How to run the programs

Create a virtual environment. (For mac users use python3 commnad)
```
python -m venv virt
```

Install libraries (pandas, matplotlib, scikit learn, seaborn and sentece transformers)
```
pip install pandas

python -m pip install -U matplotlib

pip intall -U scikit-learn

pip install seaborn

pip install sentence-transformers

```

Any question please contact by email: pedro.fonseca01@utrgv.edu




