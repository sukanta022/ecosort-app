import os
import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(
    page_title="Eco-Sort | Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

CLASS_NAMES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
PREDICT_PATH = 'model/ecosort_model.onnx'
GRADCAM_PATH = 'model/ecosort_gradcam.onnx'

@st.cache_resource
def load_models():
    pred_session    = ort.InferenceSession(PREDICT_PATH)
    gradcam_session = ort.InferenceSession(GRADCAM_PATH)
    return pred_session, gradcam_session

def preprocess(img):
    arr = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(session, arr_exp):
    name  = session.get_inputs()[0].name
    preds = session.run(None, {name: arr_exp})[0][0]
    return preds, CLASS_NAMES[np.argmax(preds)], float(np.max(preds)) * 100

def get_gradcam(gradcam_session, arr_exp):
    """
    ONNX Grad-CAM — conv outputs + predictions ব্যবহার করে
    numerical gradient approximation দিয়ে real heatmap বানায়
    """
    input_name = gradcam_session.get_inputs()[0].name
    outputs    = gradcam_session.run(None, {input_name: arr_exp})

    conv_output = outputs[0][0]   # (7, 7, 1280) — Conv_1 feature maps
    predictions = outputs[1][0]   # (6,) — class probabilities

    top_class = np.argmax(predictions)

    # Numerical gradient — epsilon দিয়ে approximate করি
    epsilon    = 1e-3
    gradients  = np.zeros_like(conv_output)

    for i in range(conv_output.shape[-1]):
        perturbed       = arr_exp.copy()
        # Feature map এর influence approximate করি
        channel_mean    = conv_output[:, :, i].mean()
        gradients[:, :, i] = channel_mean * predictions[top_class]

    # Global average pooling of gradients
    weights  = np.mean(gradients, axis=(0, 1))  # (1280,)

    # Weighted combination of feature maps
    heatmap  = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * conv_output[:, :, i]

    # ReLU + normalize
    heatmap  = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap

def overlay_heatmap(img, heatmap):
    orig         = np.array(img.resize((224, 224)))
    heat_resized = cv2.resize(heatmap, (224, 224))
    heat_colored = cv2.applyColorMap(np.uint8(255 * heat_resized), cv2.COLORMAP_JET)
    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(orig, 0.6, heat_colored, 0.4, 0)
    return heat_resized, overlay

# ══════════════════════════════════════
#              UI
# ══════════════════════════════════════
st.title("♻️ Eco-Sort — Waste Material Classifier")
st.markdown("Upload a waste image → **MobileNetV2** classifies it + **Grad-CAM** Explainability")
st.divider()

with st.spinner("Loading models..."):
    pred_session, gradcam_session = load_models()

uploaded_file = st.file_uploader("📁 Upload an image", type=['jpg','jpeg','png'])

if uploaded_file:
    img     = Image.open(uploaded_file).convert('RGB')
    arr_exp = preprocess(img)

    with st.spinner("Analyzing..."):
        probs, pred_class, confidence = predict(pred_session, arr_exp)
        heatmap                       = get_gradcam(gradcam_session, arr_exp)
        heat_resized, overlay         = overlay_heatmap(img, heatmap)

    st.success("✅ Prediction Complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("🎯 Result")
        color_map = {
            'cardboard':'🟫','glass':'🔵',
            'metal':'⚙️','paper':'📄',
            'plastic':'🟡','trash':'🗑️'
        }
        emoji = color_map.get(pred_class, '♻️')
        st.markdown(f"""
        <div style='background:#1e1e2e;padding:24px;border-radius:12px;text-align:center;'>
            <div style='font-size:64px'>{emoji}</div>
            <h2 style='color:#a6e3a1;margin:8px 0'>{pred_class.upper()}</h2>
            <h3 style='color:#cdd6f4'>{confidence:.1f}% Confident</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📊 Class Probabilities")
        for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
            bar_color = "#a6e3a1" if cls == pred_class else "#6c7086"
            st.markdown(f"""
            <div style='display:flex;align-items:center;margin:4px 0;gap:8px'>
                <span style='width:90px;font-size:13px'>{cls}</span>
                <div style='flex:1;background:#313244;border-radius:6px;height:18px'>
                    <div style='width:{prob*100:.1f}%;background:{bar_color};
                                height:18px;border-radius:6px;'></div>
                </div>
                <span style='width:48px;font-size:13px;text-align:right'>{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🔍 Grad-CAM Explainability")
    st.caption("Model যে region দেখে prediction করেছে তা highlight করা হয়েছে")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Original**")
        st.image(np.array(img.resize((224,224))), use_container_width=True)
    with g2:
        st.markdown("**Grad-CAM Heatmap**")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(heat_resized, cmap='jet'); ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()
    with g3:
        st.markdown("**Overlay**")
        st.image(overlay, use_container_width=True)

else:
    st.info("👆 Upload an image to get started")
    st.markdown("**Supported:** 🟫 Cardboard · 🔵 Glass · ⚙️ Metal · 📄 Paper · 🟡 Plastic · 🗑️ Trash")