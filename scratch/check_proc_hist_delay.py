import os
import pandas as pd

proc_path = "data/processed/multimodal_train.csv"
df = pd.read_csv(proc_path)
air = df[df['transport_mode'] == 'AIRPLANE']

print("=== AIRPLANE historical_delay_mean in multimodal_train.csv ===")
print("By Carrier:")
print(air.groupby('carrier_or_route')['historical_delay_mean'].agg(['count', 'min', 'max', 'mean']))

print("\nBy Origin (sample 5):")
print(air.groupby('origin')['historical_delay_mean'].agg(['count', 'min', 'max', 'mean']).head(5))

print("\nDistribution of historical_delay_mean:")
print(air['historical_delay_mean'].describe())
