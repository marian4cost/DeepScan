import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# ============================================================
# GLCM de uma imagem (4 metricas)
# ============================================================
def glcm_metrics(image):
    img = (image * 255).astype(np.uint8)

    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
    }

# ============================================================
# COMPARACAOO ENTRE ORIGINAL E UMA ADAPTADA
# ============================================================
def glcm_compare(original, adapted, nome="Imagem"):

    glcm_orig = glcm_metrics(original)
    glcm_adapt = glcm_metrics(adapted)

    df_orig = pd.DataFrame(glcm_orig, index=["Original"])
    df_adapt = pd.DataFrame(glcm_adapt, index=[nome])

    variance_df = (df_adapt.values - df_orig.values)
    variance_df = pd.DataFrame(variance_df, columns=df_orig.columns, index=[nome]).abs()

    return df_orig, df_adapt, variance_df

# ============================================================
# COMPARACAO COM MULTIPLAS IMAGENS
# ============================================================
def glcm_compare_multiple(original, imagens_dict):
    resultados = {}

    for nome, img in imagens_dict.items():
        df_orig, df_adapt, df_var = glcm_compare(original, img, nome)
        resultados[nome] = {
            "original": df_orig,
            "adaptada": df_adapt,
            "variancia": df_var
        }

    return resultados

# ============================================================
# EXEMPLO DE USO (5 IMAGENS)
# ============================================================

# Imagem original
original = cv2.imread("/aumentoContraste/wavelet/no-tumor01-clahe.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

# 4 imagens adaptadas
imagens = {
    "DIP_Antigo": cv2.imread("/redesNeural/wavelet/no-tumor01/dip-antigo-no-tumor01.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255,
    "DIP_Novo": cv2.imread("/redesNeural/wavelet/no-tumor01/dip-novo-no-tumor01.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255,
    "DCE_Antigo": cv2.imread("/redesNeural/wavelet/no-tumor01/dce-antigo-no-tumor01.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255,
    "DCE_Novo": cv2.imread("/redesNeural/wavelet/no-tumor01/dce-novo-no-tumor01.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255,
}

# Comparacao
resultados = glcm_compare_multiple(original, imagens)

# ============================================================
# IMPRESSAO DOS RESULTADOS
# ============================================================

for nome, res in resultados.items():
    print(f"\n==============================")
    print(f"Comparação: Original vs {nome}")
    print(f"==============================")

    print("\n--- GLCM Original ---\n")
    print(res["original"])

    print(f"\n--- GLCM {nome} ---\n")
    print(res["adaptada"])

    print(f"\n--- VARIAÇÃO (Original vs {nome}) ---\n")
    print(res["variancia"])