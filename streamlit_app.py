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
    jet = cm.colormaps["jet"]
    jet_heatmap = jet(heatmap_resized)[:, :, :3]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    overlay = np.uint8(jet_heatmap * alpha + raw_image * (1 - alpha))
    return overlay


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f6f8f4; }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #2d5a3d 0%, #4a8b5c 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem; }

    .stat-row { display: flex; gap: 1rem; margin-top: 1.2rem; }
    .stat-pill {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        flex: 1;
    }
    .stat-pill .label { font-size: 0.75rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.04em; }
    .stat-pill .value { font-size: 1.3rem; font-weight: 700; }

    .card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e8ece6;
    }

    .result-banner {
        background: linear-gradient(135deg, #4a8b5c 0%, #6bab7c 100%);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        color: white;
        margin-bottom: 1rem;
    }
    .result-banner .tag { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85; }
    .result-banner .cls { font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
    .result-banner .conf { font-size: 0.95rem; opacity: 0.9; }

    .score-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.35rem 0; }
    .score-name { width: 190px; font-size: 0.88rem; color: #333; flex-shrink: 0; }
    .score-track { flex: 1; background: #eef1ec; border-radius: 6px; height: 10px; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #4a8b5c, #7bc491); }
    .score-pct { width: 52px; text-align: right; font-size: 0.85rem; color: #555; }

    div[data-testid="stFileUploader"] { background: white; border-radius: 12px; padding: 0.5rem; border: 1px solid #e8ece6; }
</style>
""", unsafe_allow_html=True)

model, meta, is_cnn = load_deployed_model()

st.markdown(f"""
<div class="hero">
    <h1>🌿 Plant Disease Detection</h1>
    <p>Upload a leaf photo to identify the disease and see where the model focused.</p>
    <div class="stat-row">
        <div class="stat-pill"><div class="label">Model</div><div class="value">{meta['model_name']}</div></div>
        <div class="stat-pill"><div class="label">Accuracy</div><div class="value">{meta['metrics']['accuracy']*100:.1f}%</div></div>
        <div class="stat-pill"><div class="label">F1 score</div><div class="value">{meta['metrics']['f1_score']*100:.1f}%</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image_size = tuple(meta["image_size"])
    class_names = meta["class_names"]
    preprocess_fn = PREPROCESS_FNS[meta["preprocessing"]]

    pil_image = Image.open(uploaded_file).convert("RGB").resize(image_size)
    raw_array = np.array(pil_image).astype("uint8")
    img_array = tf.expand_dims(tf.convert_to_tensor(raw_array, dtype=tf.float32), 0)
    preprocessed = preprocess_fn(img_array)

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Uploaded image**")
        st.image(pil_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if is_cnn:
        heatmap, predictions = make_gradcam_heatmap(preprocessed, model)
        overlay = overlay_gradcam(raw_array, heatmap, image_size)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Grad-CAM — where the model looked**")
            st.image(overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        predictions = model.predict(preprocessed, verbose=0)[0]
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("Grad-CAM isn't available for ViT — it needs attention-rollout instead.")
            st.markdown('</div>', unsafe_allow_html=True)

    scores = {name: float(p) for name, p in zip(class_names, predictions)}
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    top_class = next(iter(sorted_scores))
    top_conf = sorted_scores[top_class]

    st.write("")
    st.markdown(f"""
    <div class="result-banner">
        <div class="tag">Predicted disease</div>
        <div class="cls">{top_class.replace('_', ' ')}</div>
        <div class="conf">{top_conf*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    max_score = max(sorted_scores.values()) or 1.0
    for name, score in sorted_scores.items():
        width_pct = (score / max_score) * 100
        rows_html += f"""
        <div class="score-row">
            <div class="score-name">{name.replace('_', ' ')}</div>
            <div class="score-track"><div class="score-fill" style="width:{width_pct:.1f}%"></div></div>
            <div class="score-pct">{score*100:.1f}%</div>
        </div>
        """

    st.markdown(f"""
    <div class="card">
        <div style="font-weight:600; margin-bottom:0.8rem;">All class scores</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem 1.5rem; color: #888;">
        Upload a leaf image above to get a prediction.
    </div>
    """, unsafe_allow_html=True)

