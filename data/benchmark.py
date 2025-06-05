import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------
# Datos extraídos de los resultados obtenidos en los entrenos
# -------------------------------------------------------------
data = {
    "Técnica": ["LoRA", "QLoRA", "Prefix Tuning", "IA3"],
    "Train Loss": [0.22262896876181326, 0.2501811088424131, 0.5425049776380713, 0.39705897089422937],
    "Eval Loss": [0.0738649070262909, 0.07818546891212463, 0.16354696452617645, 0.1521778404712677],
    "GPU Util (%)": [92.61, 91.62, 92.84, 93.38],
    "CPU Util (%)": [5.0, 5.0, 4.99, 5.0],
    "RAM Used (MB)": [832.37, 660.63, 797.1, 819.31],
    "Train Time (s)": [2990.24, 2874.11, 11150.35, 11275.45],
}

df = pd.DataFrame(data)

# -------------------------------------------------------------
# Función para convertir segundos a h:m:s (string)
# -------------------------------------------------------------
def seconds_to_hms(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:d}h {m:02d}m {s:02d}s"

df["Train Time (h:m:s)"] = df["Train Time (s)"].apply(seconds_to_hms)

# -------------------------------------------------------------
# 1) Gráfico: Train Loss vs Eval Loss
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
indices = range(len(df))
bar_width = 0.35

plt.bar(
    [i - bar_width/2 for i in indices],
    df["Train Loss"],
    width=bar_width,
    label="Train Loss",
    color="tab:blue"
)

for i, loss_val in enumerate(df["Train Loss"]):
    plt.text(
        i - bar_width/2,
        loss_val + 0.01,
        f"{loss_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
    )

plt.bar(
    [i + bar_width/2 for i in indices],
    df["Eval Loss"],
    width=bar_width,
    label="Eval Loss",
    color="tab:orange"
)

for i, loss_val in enumerate(df["Eval Loss"]):
    plt.text(
        i + bar_width/2,
        loss_val + 0.01,
        f"{loss_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
    )

plt.xticks(indices, df["Técnica"])
plt.xlabel("Técnica PEFT")
plt.ylabel("Loss")
plt.title("Train Loss vs Eval Loss (GPT-2)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2.1) Tiempo de Entrenamiento (subplot 1)
axes[0].bar(df["Técnica"], df["Train Time (s)"], color="tab:orange")
for i, (t, label) in enumerate(zip(df["Train Time (s)"], df["Train Time (h:m:s)"])):
    axes[0].text(
        i,
        t + max(df["Train Time (s)"]) * 0.01,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
    )
axes[0].set_xlabel("Técnica PEFT")
axes[0].set_ylabel("Tiempo de Entrenamiento (s)")
axes[0].set_title("Tiempo de Entrenamiento (GPT-2)")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# 2.2) Utilización de GPU y CPU (subplot 2)
indices = range(len(df))
bar_width = 0.35
axes[1].bar(
    [i - bar_width / 2 for i in indices],
    df["GPU Util (%)"],
    width=bar_width,
    label="GPU Util (%)",
    color="tab:green",
)
axes[1].bar(
    [i + bar_width / 2 for i in indices],
    df["CPU Util (%)"],
    width=bar_width,
    label="CPU Util (%)",
    color="tab:purple",
)
axes[1].set_xticks(indices)
axes[1].set_xticklabels(df["Técnica"])
axes[1].set_xlabel("Técnica PEFT")
axes[1].set_ylabel("Utilización (%)")
axes[1].set_title("Utilización de GPU y CPU (GPT-2)")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

for i in indices:
    gpu_val = df.loc[i, "GPU Util (%)"]
    cpu_val = df.loc[i, "CPU Util (%)"]
    axes[1].text(
        i - bar_width / 2,
        gpu_val + 1,
        f"{gpu_val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axes[1].text(
        i + bar_width / 2,
        cpu_val + 1,
        f"{cpu_val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# 2.3) Uso de RAM (subplot 3)
axes[2].bar(df["Técnica"], df["RAM Used (MB)"], color="tab:blue")
for i, ram in enumerate(df["RAM Used (MB)"]):
    axes[2].text(
        i,
        ram + max(df["RAM Used (MB)"]) * 0.01,
        f"{int(ram):,}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
axes[2].set_xlabel("Técnica PEFT")
axes[2].set_ylabel("RAM Utilizada (MB)")
axes[2].set_title("Uso de RAM (GPT-2)")
axes[2].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
