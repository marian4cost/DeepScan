import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------------
# UTILITÁRIOS
# -------------------------------------------------------

def pad_to_multiple(img, multiple=8):
    h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    padded = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )
    return padded, pad_h, pad_w

def remove_padding(img, pad_h, pad_w):
    if pad_h > 0:
        img = img[:-pad_h, :]
    if pad_w > 0:
        img = img[:, :-pad_w]
    return img

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img, pad_h, pad_w = pad_to_multiple(img)
    return img, pad_h, pad_w

def show_result(original, output):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("DIP – Realce Estrutural")
    plt.imshow(output, cmap="gray")
    plt.axis("off")
    plt.show()

# -------------------------------------------------------
# MODELO DIP (U-Net simplificada)
# -------------------------------------------------------

def conv_block(x, filters):
    init = tf.keras.initializers.RandomNormal(0.0, 0.01)

    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer=init
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    return x

def build_DIP():
    inp = tf.keras.Input(shape=(None, None, 1))

    e1 = conv_block(inp, 32)
    p1 = tf.keras.layers.AveragePooling2D(2)(e1)

    e2 = conv_block(p1, 64)
    p2 = tf.keras.layers.AveragePooling2D(2)(e2)

    e3 = conv_block(p2, 128)

    u2 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(e3)
    u2 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(u2)
    u1 = conv_block(u1, 32)

    out = tf.keras.layers.Conv2D(1, 1)(u1)

    return tf.keras.Model(inp, out)

# -------------------------------------------------------
# TREINAMENTO DIP
# -------------------------------------------------------

def run_DIP(img, pad_h, pad_w, iters=1000):
    h, w = img.shape

    model = build_DIP()
    opt = tf.keras.optimizers.Adam(1e-3)

    z = np.random.randn(1, h, w, 1).astype(np.float32)
    target = img.reshape(1, h, w, 1)

    best_loss = np.inf
    patience, counter = 200, 0

    for i in range(iters):
        with tf.GradientTape() as tape:
            out = model(z, training=True)
            loss = tf.reduce_mean((out - target)**2)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if loss < best_loss:
            best_loss = loss
            best_out = out.numpy()
            counter = 0
        else:
            counter += 1
            if counter > patience:
                break

        if i % 200 == 0:
            print(f"[DIP] Iter {i} | Loss {loss.numpy():.6f}")

    result = best_out[0, :, :, 0]
    return remove_padding(result, pad_h, pad_w)

# -------------------------------------------------------
# EXECUÇÃO
# -------------------------------------------------------

img, ph, pw = load_image("P_1.jpg")
dip_result = run_DIP(img, ph, pw)

show_result(remove_padding(img, ph, pw), dip_result)