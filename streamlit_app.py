
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from tensorflow.keras.models import load_model


# =========================
# CONFIG
# =========================

IMG_SIZE = (128,128)

CLASS_NAMES = ['female', 'male']


# =========================
# LOAD MODEL
# =========================

model = load_model("model.keras")


# =========================
# FUNCTIONS
# =========================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def make_prediction(img_array):

    prediction = model.predict(img_array)[0][0]

    male_prob = float(prediction)
    female_prob = 1 - male_prob

    predicted_class = CLASS_NAMES[int(prediction >= 0.5)]

    return predicted_class, male_prob, female_prob


# =========================
# SALIENCY MAP
# =========================

def generate_saliency(img_array):

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:

        tape.watch(img_tensor)

        prediction = model(img_tensor)

        loss = prediction[:,0]

    grads = tape.gradient(loss, img_tensor)

    saliency = tf.reduce_max(
        tf.abs(grads),
        axis=-1
    )[0]

    return saliency.numpy()


# =========================
# GRAD-CAM
# =========================

last_conv_layer_name = "conv2d_2"

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.outputs[0]
    ]
)


def generate_gradcam(img_array):

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor)

        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)

    # Evitar gradientes None
    if grads is None:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1]))

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_heat = tf.math.reduce_max(heatmap)

    if max_heat == 0:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1]))

    heatmap /= max_heat

    return heatmap.numpy()

# =========================
# STREAMLIT UI
# =========================

st.title("Gender Classification CNN")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    predicted_class, male_prob, female_prob = make_prediction(img_array)

    st.subheader("Prediction")

    st.write(f"Predicted Class: {predicted_class}")

    st.write(f"Male Probability: {male_prob:.4f}")

    st.write(f"Female Probability: {female_prob:.4f}")


    # =========================
    # SALIENCY MAP
    # =========================

    saliency = generate_saliency(img_array)

    fig1, ax1 = plt.subplots()

    ax1.imshow(img_array[0])

    ax1.imshow(saliency, cmap='hot', alpha=0.5)

    ax1.set_title("Saliency Map")

    ax1.axis("off")

    st.pyplot(fig1)


    # =========================
    # GRAD-CAM
    # =========================

    heatmap = generate_gradcam(img_array)

    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE[1], IMG_SIZE[0])
    )

    fig2, ax2 = plt.subplots()

    ax2.imshow(img_array[0])

    ax2.imshow(heatmap, cmap='jet', alpha=0.5)

    ax2.set_title("Grad-CAM")

    ax2.axis("off")

    st.pyplot(fig2)
