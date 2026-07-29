"""
Plant Disease Detection — Streamlit App
========================================

Loads the deployment package produced by the notebook (Stage 10):
    deployment/
      model.keras
      metadata.json

Deploy for free at https://share.streamlit.io by connecting this repo.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image
import matplotlib.cm as cm


# --------------------------------------------------------------------------
# Custom layers needed to load the ViT model (harmless to import if unused)
# --------------------------------------------------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config


PREPROCESS_FNS = {
    "resnet50": resnet_preprocess,
    "efficientnet": efficientnet_preprocess,
    "rescale": lambda x: x / 255.0,
}


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session, not per click)
# --------------------------------------------------------------------------
@st.cache_resource
def load_deployed_model(deployment_dir="deployment"):
    with open(os.path.join(deployment_dir, "metadata.json")) as f:
        meta = json.load(f)

    custom_objects = None
    if meta.get("custom_objects"):
        custom_objects = {"Patches": Patches, "PatchEncoder": PatchEncoder}

    model = tf.keras.models.load_model(
        os.path.join(deployment_dir, "model.keras"), custom_objects=custom_objects
    )
    is_cnn = meta["model_name"] in ("resnet50", "efficientnetb0")
    return model, meta, is_cnn


# --------------------------------------------------------------------------
# Grad-CAM (only used when the deployed model is a CNN)
# --------------------------------------------------------------------------
def find_last_conv_layer(backbone):
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No 4D conv layer found in this backbone.")


def make_gradcam_heatmap(img_array, full_model):
    backbone = full_model.layers[0]
    last_conv_layer_name = find_last_conv_layer(backbone)

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(img_array)
        x = backbone_output
        for layer in full_model.layers[1:]:
            x = layer(x, training=False)
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()[0]


def overlay_gradcam(raw_image, heatmap, image_size, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = tf.image.resize(
        heatmap_uint8[..., np.newaxis], image_size
    ).numpy().astype("uint8")[..., 0]
    jet = cm.get_cmap("jet")
    jet_heatmap = jet(heatmap_resized)[:, :, :3]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    overlay = np.uint8(jet_heatmap * alpha + raw_image * (1 - alpha))
    return overlay


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

model, meta, is_cnn = load_deployed_model()

st.title("🌿 Plant Disease Detection")
st.caption(
    f"Serving **{meta['model_name']}** — "
    f"val accuracy: {meta['metrics']['accuracy']:.3f}, "
    f"F1: {meta['metrics']['f1_score']:.3f}"
)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_size = tuple(meta["image_size"])
    class_names = meta["class_names"]
    preprocess_fn = PREPROCESS_FNS[meta["preprocessing"]]

    pil_image = Image.open(uploaded_file).convert("RGB").resize(image_size)
    raw_array = np.array(pil_image).astype("uint8")
    img_array = tf.expand_dims(tf.convert_to_tensor(raw_array, dtype=tf.float32), 0)
    preprocessed = preprocess_fn(img_array)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    if is_cnn:
        heatmap, predictions = make_gradcam_heatmap(preprocessed, model)
        overlay = overlay_gradcam(raw_array, heatmap, image_size)
        with col2:
            st.image(overlay, caption="Grad-CAM", use_container_width=True)
    else:
        predictions = model.predict(preprocessed, verbose=0)[0]
        with col2:
            st.info("Grad-CAM isn't available for ViT — it needs attention-rollout instead.")

    scores = {name: float(p) for name, p in zip(class_names, predictions)}
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    st.subheader("Prediction")
    top_class = next(iter(sorted_scores))
    st.success(f"**{top_class}** ({sorted_scores[top_class]*100:.1f}% confidence)")

    st.subheader("All class scores")
    st.bar_chart(sorted_scores)
