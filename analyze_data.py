import pandas as pd

df = pd.read_csv("data.csv")

print("=" * 50)
print("SHAPE")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("HEAD")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("DESCRIBE")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("DTYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("DUPLICATES")
print("=" * 50)
print(df.duplicated().sum())

print("\n" + "=" * 50)
print("LABEL DISTRIBUTION")
print("=" * 50)
print(df["label"].value_counts())

print("\n" + "=" * 50)
print("CORRELATION")
print("=" * 50)
print(df.corr())

print("\n" + "=" * 50)
print("STATS BY LABEL")
print("=" * 50)
print(df.groupby("label").describe())
