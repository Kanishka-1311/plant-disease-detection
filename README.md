# 🌿 Plant Disease Detection

A deep learning project that classifies plant leaf diseases from an image across
9 disease classes (Apple, Corn, Grape, Potato, Tomato). Three architectures —
ResNet50, EfficientNetB0, and a Vision Transformer (ViT) — were trained and
compared, with the best model (ResNet50) deployed as a live web app.

**🔗 Live app:** https://plant-disease-detection-kanishka.streamlit.app

The app includes Grad-CAM visualization, so you can see which part of the leaf
the model focused on to make its prediction.

## Model comparison

| Model | Accuracy | F1 Score |
|---|---|---|
| **ResNet50 (deployed)** | 0.9097 | 0.9103 |
| EfficientNetB0 | 0.8957 | 0.8957 |
| ViT (trained from scratch) | 0.6145 | 0.6008 |

ResNet50 and EfficientNetB0 use ImageNet transfer learning; ViT was trained
from scratch, which explains its lower score on a relatively small dataset.

## Repo structure
```
.
├── streamlit_app.py       # the web app (Streamlit)
├── requirements.txt       # Python dependencies
└── deployment/
    ├── model.keras         # the trained ResNet50 model
    └── metadata.json       # class names, image size, preprocessing config
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open the local URL it prints (e.g. http://localhost:8501).

## Tech stack
Python · TensorFlow / Keras · Streamlit · Grad-CAM · scikit-learn
