import os
import streamlit as st
import numpy as np
import onnxruntime as ort
import tensorflow as tf
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
ONNX_PATH   = 'model/ecosort_model.onnx'   # predict
TF_PATH     = 'model/ecosort_phase1.h5'    # grad-cam

@st.cache_resource
def load_onnx():
    return ort.InferenceSession(ONNX_PATH)

@st.cache_resource
def load_tf():
    return tf.keras.models.load_model(TF_PATH)

def predict(session, img):
    arr        = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    arr_exp    = np.expand_dims(arr, 0)
    input_name = session.get_inputs()[0].name
    preds      = session.run(None, {input_name: arr_exp})[0][0]
    return preds, CLASS_NAMES[np.argmax(preds)], float(np.max(preds))*100

def get_gradcam(model, img, layer_name='Conv_1'):
    arr     = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    arr_exp = np.expand_dims(arr, 0)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr_exp)
        top_class = tf.argmax(preds[0])
        loss = preds[:, top_class]

    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    orig         = np.array(img.resize((224,224)))
    heat_resized = cv2.resize(heatmap, (224,224))
    heat_colored = cv2.applyColorMap(np.uint8(255*heat_resized), cv2.COLORMAP_JET)
    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(orig, 0.6, heat_colored, 0.4, 0)

    return heat_resized, overlay

# ══════════════════════════════════════
#              UI
# ══════════════════════════════════════
st.title("♻️ Eco-Sort — Waste Material Classifier")
st.markdown("Upload a waste image → **MobileNetV2** classifies it + **Real Grad-CAM** Explainability")
st.divider()

with st.spinner("Loading models..."):
    onnx_session = load_onnx()
    tf_model     = load_tf()

uploaded_file = st.file_uploader("📁 Upload an image", type=['jpg','jpeg','png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    with st.spinner("Analyzing..."):
        probs, pred_class, confidence = predict(onnx_session, img)
        heatmap, overlay = get_gradcam(tf_model, img)

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
    st.subheader("🔍 Grad-CAM — Real Explainability")
    st.caption("Model যে region দেখে prediction করেছে তা highlight করা হয়েছে")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("**Original**")
        st.image(np.array(img.resize((224,224))), use_column_width=True)

    with g2:
        st.markdown("**Grad-CAM Heatmap**")
        fig, ax = plt.subplots(figsize=(3,3))
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
    st.info("👆 Upload an image to get started")
    st.markdown("**Supported:** 🟫 Cardboard · 🔵 Glass · ⚙️ Metal · 📄 Paper · 🟡 Plastic · 🗑️ Trash")