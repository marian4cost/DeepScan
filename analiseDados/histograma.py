import cv2
import matplotlib.pyplot as plt

def gerar_histograma(path_imagem):
    # Carrega em escala de cinza
    img = cv2.imread(path_imagem, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Erro: imagem não encontrada ou caminho inválido.")

    # Cria o histograma
    plt.figure(figsize=(6, 5))
    plt.hist(img.ravel(), bins=256, color='gray')
    plt.title("Histograma da Imagem")
    plt.xlabel("Intensidade (0-255)")
    plt.ylabel("Frequência")
    plt.show()


# >>> coloque o caminho da sua imagem aqui <<<
caminho = "dceNovo.png"

# Chama a função
gerar_histograma(caminho)
