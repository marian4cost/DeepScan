import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Adição da biblioteca para gráficos estatísticos

# ======================================================
# 1. MODELO DE DATAFRAME (Mantido conforme seu original)
# ======================================================
imagens = ["img1", "img2", "img3", "img4", "img5"]
tecnicas = ["DIP Antigo", "DIP Novo", "DCE Antigo", "DCE Novo"]

linhas = []
for img in imagens:
    for tec in tecnicas:
        linhas.append([img, tec])

df = pd.DataFrame(linhas, columns=["Imagem", "Tecnica"])

df["Contrast"] = [
    59.01, 59.56, 79.18, 71.91,
    155.04, 148.31, 94.45, 156.66,
    750.23, 736.84, 575.08, 673.14,
    210.56, 216.51, 237.21, 182.95,
    377.32, 352.89, 322.82, 415.26,
]
df["Homogeneity"] = [
    25.99, 5.77, 95.71, 21.85,
    14.96, 2.01, 109.24, 14.12,
    47.43, 34.75, 20.41, 9.08,
    29.29, 1.75, 50.95, 37.13,
    46.33, 35.94, 90.37, 12.20,
]
df["Energy"] = [
    31.41, 15.32, 553.48, 4.43,
    2.16, 4.04, 986.18, 92.63,
    62.81, 69.93, 96.65, 49.90,
    58.29, 18.73, 140.44, 95.28,
    57.84, 58.57, 1875.86, 74.85,
]
df["Correlation"] = [
    0.38, 0.31, 8.16, 0.52,
    0.77, 0.73, 9.81, 0.15,
    1.90, 2.03, 14.11, 0.55,
    1.18, 1.16, 8.23, 0.33,
    1.91, 1.87, 12.84, 1.05,
]

# ======================================================
# 2. PREPARAÇÃO DOS DADOS PARA O VIOLIN PLOT
# ======================================================
# Para o Violin Plot, precisamos transformar o DF para o formato "long" (long-form)
df_melted = df.melt(id_vars=["Imagem", "Tecnica"], 
                    var_name="Metrica", 
                    value_name="Valor")

# ======================================================
# 3. GERAÇÃO DO VIOLIN PLOT
# ======================================================
plt.figure(figsize=(14, 8))

# Criando o Violin Plot
# Ele mostrará a distribuição de TODAS as métricas agrupadas por Técnica
sns.violinplot(data=df_melted, x="Tecnica", y="Valor", hue="Tecnica", 
               palette="viridis", inner="quartile", legend=False)

# Adicionando um "Swarm Plot" por cima (opcional) para ver os pontos individuais das 5 imagens
sns.swarmplot(data=df_melted, x="Tecnica", y="Valor", color="white", edgecolor="gray", alpha=0.6, size=4)

plt.title("Gráfico de Violino - AMBAS TÉCNICAS", fontsize=16)
plt.xlabel("Técnicas", fontsize=12)
plt.ylabel("Valores", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()