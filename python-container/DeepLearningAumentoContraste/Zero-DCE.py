import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------------
# UTILITÁRIOS
# -------------------------------------------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, img.shape[0], img.shape[1], 1)

def show_result(original, enhanced):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Imagem Original")
    plt.imshow(original.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Zero-DCE – Aumento de Contraste")
    plt.imshow(enhanced.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

# -------------------------------------------------------
# MODELO ZERO-DCE
# -------------------------------------------------------

def build_zero_dce():
    inp = tf.keras.Input(shape=(None, None, 1))

    x = inp
    for _ in range(7):
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    curves = tf.keras.layers.Conv2D(8, 3, padding="same", activation="tanh")(x)
    return tf.keras.Model(inp, curves)

# -------------------------------------------------------
# APLICAÇÃO DAS CURVAS
# -------------------------------------------------------

def apply_curves(img, curves):
    enhanced = img
    for i in range(curves.shape[-1]):
        enhanced = enhanced + curves[..., i:i+1] * (
            enhanced - enhanced**2
        )
    return tf.clip_by_value(enhanced, 0.0, 1.0)

# -------------------------------------------------------
# TREINAMENTO ZERO-DCE
# -------------------------------------------------------

def run_zero_dce(img, iters=1000):
    model = build_zero_dce()
    opt = tf.keras.optimizers.Adam(1e-4)

    for i in range(iters):
        with tf.GradientTape() as tape:
            curves = model(img)
            enhanced = apply_curves(img, curves)

            # Loss simples de contraste
            loss = -tf.reduce_mean(tf.image.sobel_edges(enhanced))

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if i % 100 == 0:
            print(f"[Zero-DCE] Iter {i} | Loss {loss.numpy():.6f}")

    curves = model(img)
    return apply_curves(img, curves).numpy()

# -------------------------------------------------------
# EXECUÇÃO
# -------------------------------------------------------

img = load_image("P_1.jpg")
enhanced = run_zero_dce(img)

show_result(img, enhanced)