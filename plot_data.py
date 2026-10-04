import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_scaled.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = {0: "red", 1: "green", 2: "blue"}

# Plot 1: F1 vs F2
for label in [0, 1, 2]:
    subset = df[df["label"] == label]
    axes[0].scatter(subset["F1"], subset["F2"], c=colors[label],
                    label=f"Class {label}", alpha=0.5, s=10)
axes[0].set_xlabel("F1")
axes[0].set_ylabel("F2")
axes[0].set_title("F1 vs F2")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Histogram of F1
for label in [0, 1, 2]:
    subset = df[df["label"] == label]
    axes[1].hist(subset["F1"], bins=30, alpha=0.5,
                 color=colors[label], label=f"Class {label}")
axes[1].set_xlabel("F1")
axes[1].set_ylabel("Count")
axes[1].set_title("F1 Distribution")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("plot.png", dpi=150)
print("Saved: plot.png")

# کپی به حافظه گوشی
import shutil
shutil.copy("plot.png", "/data/data/com.termux/files/home/storage/downloads/plot.png")
print("Copied to Downloads")
