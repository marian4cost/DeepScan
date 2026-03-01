from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np

# 1. Carregar imagem
img = img_as_float(io.imread("original.jpg"))

# 2. Converter pra grayscale caso seja RGB
if img.ndim == 3:
    img = rgb2gray(img)

# 3. Estimar sigma
sigma_est = estimate_sigma(img, channel_axis=None)

# Caso seja lista, tirar média
if isinstance(sigma_est, list):
    sigma = float(np.mean(sigma_est))
else:
    sigma = float(sigma_est)

# 4. Aplicar NLM
denoised = denoise_nl_means(
    img,
    h=1.15 * sigma,
    patch_size=5,
    patch_distance=6,
    channel_axis=None,
    fast_mode=True
)

# 5. Converter para uint8 e salvar
io.imsave("pituario01.png", img_as_ubyte(denoised))