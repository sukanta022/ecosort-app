import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

# ── Page Config ──
st.set_page_config(
    page_title="Eco-Sort | Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

# ── Constants ──
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH  = 'model/ecosort_model.h5'

# ── Load Model (cache করা হবে, বারবার load হবে না) ──
@st.cache_resource
def load_model_file():
    return keras.models.load_model(MODEL_PATH)

# ── Grad-CAM Function ──
def get_gradcam(model, img_array, layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        top_class = tf.argmax(preds[0])
        loss = preds[:, top_class]

    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(orig_img_rgb, heatmap):
    orig   = np.array(orig_img_rgb.resize((224, 224)))
    heat_r = cv2.resize(heatmap, (224, 224))
    heat_c = cv2.applyColorMap(np.uint8(255 * heat_r), cv2.COLORMAP_JET)
    heat_c = cv2.cvtColor(heat_c, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.6, heat_c, 0.4, 0)
    return overlay

# ── Predict Function ──
def predict(model, img):
    img_resized = img.resize((224, 224))
    arr         = np.array(img_resized) / 255.0
    arr_exp     = np.expand_dims(arr, axis=0)

    preds      = model.predict(arr_exp, verbose=0)
    pred_idx   = np.argmax(preds[0])
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(preds[0])) * 100
    return preds[0], pred_class, confidence, arr_exp

# ════════════════════════════════════════
#              UI LAYOUT
# ════════════════════════════════════════

st.title("♻️ Eco-Sort — Waste Material Classifier")
st.markdown("Upload a waste image and the model will classify it using **MobileNetV2 + Grad-CAM**")
st.divider()

# Load model
with st.spinner("Loading model..."):
    model = load_model()

# ── Upload Section ──
uploaded_file = st.file_uploader(
    "📁 Upload an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    # ── Run Prediction ──
    with st.spinner("Analyzing..."):
        probs, pred_class, confidence, arr_exp = predict(model, img)
        heatmap = get_gradcam(model, arr_exp)
        overlay = overlay_gradcam(img, heatmap)

    st.success(f"✅ Prediction Complete!")

    # ════ Row 1: Image + Result ════
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("🎯 Prediction Result")

        # Big result display
        color_map = {
            'cardboard': '🟫', 'glass': '🔵',
            'metal': '⚙️',    'paper': '📄',
            'plastic': '🟡',   'trash': '🗑️'
        }
        emoji = color_map.get(pred_class, '♻️')

        st.markdown(f"""
        <div style='background:#1e1e2e; padding:20px; border-radius:12px; text-align:center;'>
            <h1 style='font-size:60px; margin:0'>{emoji}</h1>
            <h2 style='color:#a6e3a1; margin:5px 0'>{pred_class.upper()}</h2>
            <h3 style='color:#cdd6f4'>{confidence:.1f}% Confident</h3>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("##### 📊 Class Probabilities")
        for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
            bar_color = "#a6e3a1" if cls == pred_class else "#6c7086"
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin:4px 0; gap:8px'>
                <span style='width:90px; font-size:13px'>{cls}</span>
                <div style='flex:1; background:#313244; border-radius:6px; height:18px'>
                    <div style='width:{prob*100:.1f}%; background:{bar_color};
                                height:18px; border-radius:6px;'></div>
                </div>
                <span style='width:45px; font-size:13px; text-align:right'>{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ════ Row 2: Grad-CAM ════
    st.subheader("🔍 Grad-CAM Explainability")
    st.caption("Highlighted regions show where the model focused to make its decision")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("**Original**")
        st.image(img.resize((224, 224)), use_column_width=True)

    with g2:
        st.markdown("**Grad-CAM Heatmap**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()

    with g3:
        st.markdown("**Overlay**")
        st.image(overlay, use_column_width=True)

else:
    # Placeholder when no image uploaded
    st.info("👆 Upload an image to get started")
    st.markdown("""
    **Supported classes:**
    🟫 Cardboard · 🔵 Glass · ⚙️ Metal · 📄 Paper · 🟡 Plastic · 🗑️ Trash
    """)