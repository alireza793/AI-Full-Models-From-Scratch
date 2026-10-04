import pandas as pd

df = pd.read_csv("data.csv")

print("BEFORE SCALING")
print(df.describe())

# Manual StandardScaler
for col in ["F1", "F2"]:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

print("\nAFTER SCALING")
print(df.describe())

print("\nBY LABEL")
print(df.groupby("label").mean())

df.to_csv("data_scaled.csv", index=False)
print("\nSaved to data_scaled.csv")
