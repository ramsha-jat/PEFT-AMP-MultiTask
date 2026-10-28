import pandas as pd
import numpy as np

# 1. Load EC Potency
df_ec = pd.read_csv("eval_prev_works/AMP_regression_EC_SA/data/EC.csv")
df_ec = df_ec[['SEQUENCE', 'EC_pMIC']].rename(columns={'SEQUENCE': 'sequence', 'EC_pMIC': 'potency_pmic'})

# 2. Load Hemolysis Toxicity
df_hemo = pd.read_csv("hemolysis_dataset.csv")
df_hemo = df_hemo[['sequence', 'log_hc50']].rename(columns={'log_hc50': 'toxicity_log_hc50'})

# 3. Outer Merge (Sequences with either potency, toxicity, or both)
df_multi = pd.merge(df_ec, df_hemo, on='sequence', how='outer')
print("Multi-Task Dataset Statistics:")
print(f"Total Unique Sequences: {len(df_multi)}")
print(f"Sequences with Potency: {df_multi['potency_pmic'].notnull().sum()}")
print(f"Sequences with Toxicity: {df_multi['toxicity_log_hc50'].notnull().sum()}")
print(f"Sequences with BOTH (Dual-labeled): {(df_multi['potency_pmic'].notnull() & df_multi['toxicity_log_hc50'].notnull()).sum()}")

df_multi.to_csv("multitask_dataset.csv", index=False)
print("Saved to multitask_dataset.csv")