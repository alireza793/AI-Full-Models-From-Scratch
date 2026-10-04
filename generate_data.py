import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000
x = np.random.uniform(0.1, 2000, N)

# Class A
A_F1 = np.log10(x) + x
A_F2 = np.sqrt(x) - np.log10(x)

# Class B
B_F1 = np.log(x) - np.log10(x)
B_F2 = np.cos(x) - np.log10(x)

# Class C
C_F1 = 1.2 ** (0.001 * x)
C_F2 = 1.3 * x + 9.2 * np.log(x)

df = pd.DataFrame({
    "F1": np.concatenate([A_F1, B_F1, C_F1]),
    "F2": np.concatenate([A_F2, B_F2, C_F2]),
    "label": np.concatenate([np.zeros(N), np.ones(N), np.full(N, 2)])
})

df.to_csv("data.csv", index=False)
print(df.head())
print(df.shape)
