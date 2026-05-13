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

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH  = 'model/ecosort_model.onnx'

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

def predict(session, img):
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr_exp = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr_exp})[0][0]

    pred_idx   = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(preds)) * 100
    return preds, pred_class, confidence

def get_gradcam_simple(session, img):
    """ONNX-এ GradientTape নেই, তাই occlusion-based heatmap"""
    img_resized = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    input_name  = session.get_inputs()[0].name
    patch_size  = 32
    heatmap     = np.zeros((224, 224))

    base_input = np.expand_dims(img_resized, 0)
    base_pred  = session.run(None, {input_name: base_input})[0][0]
    base_class = np.argmax(base_pred)
    base_conf  = base_pred[base_class]

    for y in range(0, 224, patch_size):
        for x in range(0, 224, patch_size):
            occluded = img_resized.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 0.5
            inp  = np.expand_dims(occluded, 0)
            pred = session.run(None, {input_name: inp})[0][0]
            drop = base_conf - pred[base_class]
            heatmap[y:y+patch_size, x:x+patch_size] = drop

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

# ══════════════════════════════════════
#            UI
# ══════════════════════════════════════
st.title("♻️ Eco-Sort — Waste Material Classifier")
st.markdown("Upload a waste image → **MobileNetV2** classifies it + **Explainability heatmap**")
st.divider()

with st.spinner("Loading model..."):
    session = load_model()

uploaded_file = st.file_uploader("📁 Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    with st.spinner("Analyzing..."):
        probs, pred_class, confidence = predict(session, img)

    st.success("✅ Prediction Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("🎯 Result")
        color_map = {
            'cardboard':'🟫','glass':'🔵',
            'metal':'⚙️','paper':'📄',
            'plastic':'🟡','trash':'🗑️'
        }
        emoji = color_map.get(pred_class, '♻️')
        st.markdown(f"""
        <div style='background:#1e1e2e;padding:20px;border-radius:12px;text-align:center;'>
            <h1 style='font-size:60px;margin:0'>{emoji}</h1>
            <h2 style='color:#a6e3a1;margin:5px 0'>{pred_class.upper()}</h2>
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
                    <div style='width:{prob*100:.1f}%;background:{bar_color};height:18px;border-radius:6px;'></div>
                </div>
                <span style='width:45px;font-size:13px;text-align:right'>{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🔍 Explainability Heatmap")

    with st.spinner("Generating heatmap (takes ~10s)..."):
        heatmap = get_gradcam_simple(session, img)

    orig = np.array(img.resize((224, 224)))
    heat_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.6, heat_colored, 0.4, 0)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Original**")
        st.image(orig, use_column_width=True)
    with g2:
        st.markdown("**Heatmap**")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(heatmap, cmap='jet'); ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0); st.image(buf, use_column_width=True); plt.close()
    with g3:
        st.markdown("**Overlay**")
        st.image(overlay, use_column_width=True)

else:
    st.info("👆 Upload an image to get started")
    st.markdown("**Supported:** 🟫 Cardboard · 🔵 Glass · ⚙️ Metal · 📄 Paper · 🟡 Plastic · 🗑️ Trash")