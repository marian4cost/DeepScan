import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1. NOVO MODELO DE DATAFRAME (20 resultados)
# ======================================================

# Estrutura:
# 5 Imagens: img1 ... img5
# 4 Técnicas: DIP Antigo, DIP Novo, DCE Antigo, DCE Novo

imagens = ["img1", "img2", "img3", "img4", "img5"]
tecnicas = ["DIP Antigo", "DIP Novo", "DCE Antigo", "DCE Novo"]

# Cria linhas combinando Imagem × Técnica
linhas = []
for img in imagens:
    for tec in tecnicas:
        linhas.append([img, tec])

df = pd.DataFrame(linhas, columns=["Imagem", "Tecnica"])

# Agora você insere os valores reais:

df["Contrast"] = [
    # img1 (DIP Antigo, DIP Novo, DCE Antigo, DCE Novo)
    52.426, 66.698, 413.013, 57.621,
    # img2
    38.774, 39.437, 653.235, 243.157,
    # img3
    234.298, 232.134, 525.894, 440.629,
    # img4
    176.357, 176.776, 359.439, 253.006,
    # img5
    228.576, 224.312, 244.057, 297.370,
]

df["Homogeneity"] = [
    # mesma estrutura
    8.916, 87.922, 516.500, 220.408,
    62.026, 152.122, 485.581, 303.656,
    108.993, 134.353, 215.928, 203.341,
    74.963, 29.158, 274.660, 306.120,
    76.382, 38.444, 512.897, 27.811,
]

df["Energy"] = [
    11.893, 36.816, 582.913, 48.756,
    18.328, 10.865, 565.807, 323.207,
    81.866, 231.331, 198.715, 194.781,
    92.775, 27.565, 307.130, 334.331,
    27.786, 25.136, 795.125, 47.667,
]

df["Correlation"] = [
    20.454, 21.881, 3.089, 29.675,
    19.923, 20.485, 37.876, 26.094,
    15.969, 17.120, 9.747, 7.136,
    9.544, 9.337, 3.170, 1.160,
    18.393, 18.748, 65.414, 6.517,
]

# ======================================================
# 2. Agora, você pode calcular o desvio padrão.
# ======================================================

# Exemplo: std por técnica
df_std = df.groupby("Tecnica")[["Contrast", "Homogeneity", "Energy", "Correlation"]].std()

# ======================================================
# 3. Gráfico igual ao seu (linhas)
# ======================================================

plt.figure(figsize=(12, 6))

for coluna in df_std.columns:
    plt.plot(df_std.index, df_std[coluna], marker='o', linewidth=2, label=coluna)

plt.title("Desvio Padrão das Métricas por Técnica (5 imagens)", fontsize=16)
plt.xlabel("Técnica", fontsize=12)
plt.ylabel("Desvio Padrão", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Métricas")
plt.tight_layout()
plt.show()