import os
import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(
    page_title="Eco-Sort | Intelligent Waste Classification & Sorting",
    page_icon="♻️",
    layout="wide"
)

# ── Constants ──
CLASS_NAMES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
PREDICT_PATH = 'model/ecosort_model.onnx'
GRADCAM_PATH = 'model/ecosort_gradcam.onnx'

# ── Sorting Framework ──
SORTING_FRAMEWORK = {
    'cardboard': {
        'bin':        'Bin A — Dry Recyclables',
        'category':   'Recyclable',
        'conveyor':   'Lane 1',
        'action':     'Flatten & Compress',
        'destination':'Paper Recycling Facility',
        'color':      '#8B6914'
    },
    'glass': {
        'bin':        'Bin B — Fragile Recyclables',
        'category':   'Recyclable',
        'conveyor':   'Lane 2',
        'action':     'Separate by Color & Crush',
        'destination':'Glass Processing Plant',
        'color':      '#1565C0'
    },
    'metal': {
        'bin':        'Bin C — Metal Recyclables',
        'category':   'Recyclable',
        'conveyor':   'Lane 3',
        'action':     'Magnetic Separation & Bale',
        'destination':'Metal Scrap Facility',
        'color':      '#37474F'
    },
    'paper': {
        'bin':        'Bin D — Dry Recyclables',
        'category':   'Recyclable',
        'conveyor':   'Lane 1',
        'action':     'Stack & Bundle',
        'destination':'Paper Recycling Facility',
        'color':      '#2E7D32'
    },
    'plastic': {
        'bin':        'Bin E — Plastic Recyclables',
        'category':   'Recyclable',
        'conveyor':   'Lane 4',
        'action':     'Shred & Pelletize',
        'destination':'Plastic Recycling Plant',
        'color':      '#F57F17'
    },
    'trash': {
        'bin':        'Bin F — General Waste',
        'category':   'Non-Recyclable',
        'conveyor':   'Lane 5',
        'action':     'Compact & Seal',
        'destination':'Controlled Landfill / Incineration',
        'color':      '#B71C1C'
    }
}

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
    input_name  = gradcam_session.get_inputs()[0].name
    outputs     = gradcam_session.run(None, {input_name: arr_exp})
    conv_output = outputs[0][0]
    predictions = outputs[1][0]
    top_class   = np.argmax(predictions)

    gradients = np.zeros_like(conv_output)
    for i in range(conv_output.shape[-1]):
        channel_mean           = conv_output[:, :, i].mean()
        gradients[:, :, i]     = channel_mean * predictions[top_class]

    weights = np.mean(gradients, axis=(0, 1))
    heatmap = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * conv_output[:, :, i]

    heatmap = np.maximum(heatmap, 0)
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

# ══════════════════════════════════════════════════════════════
#                          UI
# ══════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='text-align:center; color:#2E7D32; font-size:2.2rem; margin-bottom:0'>
    Eco-Sort
</h1>
<p style='text-align:center; color:#555; font-size:1rem; margin-top:4px'>
    Intelligent Waste Classification & Automated Sorting System
</p>
""", unsafe_allow_html=True)

st.divider()

with st.spinner("Initializing models..."):
    pred_session, gradcam_session = load_models()

uploaded_file = st.file_uploader(
    "Upload a waste image for classification and sorting decision",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    img     = Image.open(uploaded_file).convert('RGB')
    arr_exp = preprocess(img)

    with st.spinner("Running classification pipeline..."):
        probs, pred_class, confidence = predict(pred_session, arr_exp)
        heatmap                       = get_gradcam(gradcam_session, arr_exp)
        heat_resized, overlay         = overlay_heatmap(img, heatmap)

    sort_info = SORTING_FRAMEWORK[pred_class]

    # ── Section 1: Classification Result ──
    st.markdown("### Classification Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Image**")
        st.image(img, use_container_width=True)

    with col2:
        st.markdown("**Prediction**")
        st.markdown(f"""
        <div style='background:#1e1e2e; padding:24px; border-radius:10px;
                    border-left: 5px solid {sort_info["color"]}; margin-bottom:16px'>
            <div style='color:#aaa; font-size:0.8rem; text-transform:uppercase;
                        letter-spacing:1px'>Detected Material</div>
            <div style='color:#fff; font-size:2rem; font-weight:700;
                        margin:6px 0'>{pred_class.upper()}</div>
            <div style='color:#aaa; font-size:0.9rem'>Confidence: 
                <span style='color:#a6e3a1; font-weight:600'>{confidence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Class Probability Distribution**")
        for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
            bar_color = sort_info["color"] if cls == pred_class else "#444"
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin:5px 0; gap:10px'>
                <span style='width:85px; font-size:12px; color:#ccc'>{cls}</span>
                <div style='flex:1; background:#2a2a3e; border-radius:4px; height:16px'>
                    <div style='width:{prob*100:.1f}%; background:{bar_color};
                                height:16px; border-radius:4px;'></div>
                </div>
                <span style='width:44px; font-size:12px; color:#ccc;
                             text-align:right'>{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Section 2: Sorting Decision ──
    st.markdown("### Automated Sorting Decision")
    st.markdown(
        "Based on the classification result, the sorting controller routes "
        "the item through the appropriate processing pipeline:"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div style='background:#1e1e2e; padding:16px; border-radius:8px;
                    border-top:3px solid {sort_info["color"]}; height:120px'>
            <div style='color:#888; font-size:11px; text-transform:uppercase;
                        letter-spacing:1px'>Assigned Bin</div>
            <div style='color:#fff; font-size:14px; font-weight:600;
                        margin-top:8px'>{sort_info["bin"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style='background:#1e1e2e; padding:16px; border-radius:8px;
                    border-top:3px solid {sort_info["color"]}; height:120px'>
            <div style='color:#888; font-size:11px; text-transform:uppercase;
                        letter-spacing:1px'>Conveyor Lane</div>
            <div style='color:#fff; font-size:14px; font-weight:600;
                        margin-top:8px'>{sort_info["conveyor"]}</div>
            <div style='color:#aaa; font-size:12px;
                        margin-top:4px'>Category: {sort_info["category"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style='background:#1e1e2e; padding:16px; border-radius:8px;
                    border-top:3px solid {sort_info["color"]}; height:120px'>
            <div style='color:#888; font-size:11px; text-transform:uppercase;
                        letter-spacing:1px'>Processing Action</div>
            <div style='color:#fff; font-size:14px; font-weight:600;
                        margin-top:8px'>{sort_info["action"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style='background:#1e1e2e; padding:16px; border-radius:8px;
                    border-top:3px solid {sort_info["color"]}; height:120px'>
            <div style='color:#888; font-size:11px; text-transform:uppercase;
                        letter-spacing:1px'>Destination</div>
            <div style='color:#fff; font-size:14px; font-weight:600;
                        margin-top:8px'>{sort_info["destination"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sorting Pipeline Diagram ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Sorting Pipeline — Active Route**")

    all_lanes = {
        'Lane 1': ['cardboard', 'paper'],
        'Lane 2': ['glass'],
        'Lane 3': ['metal'],
        'Lane 4': ['plastic'],
        'Lane 5': ['trash'],
    }

    lane_cols = st.columns(5)
    for idx, (lane, materials) in enumerate(all_lanes.items()):
        is_active = sort_info['conveyor'] == lane
        bg        = sort_info['color'] if is_active else '#2a2a3e'
        txt_color = '#fff' if is_active else '#555'
        border    = f'2px solid {sort_info["color"]}' if is_active else '2px solid #333'

        with lane_cols[idx]:
            st.markdown(f"""
            <div style='background:{bg}; border:{border}; border-radius:8px;
                        padding:12px; text-align:center'>
                <div style='color:{txt_color}; font-size:11px;
                            font-weight:600'>{lane}</div>
                <div style='color:{txt_color}; font-size:10px;
                            margin-top:4px'>{" / ".join(m.capitalize() for m in materials)}</div>
                {'<div style="color:#fff; font-size:10px; margin-top:6px; font-weight:700">ACTIVE</div>' if is_active else ''}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Section 3: Grad-CAM ──
    st.markdown("### Grad-CAM Explainability")
    st.markdown(
        "The heatmap highlights the spatial regions of the input image "
        "that most influenced the model's classification decision."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Original Input**")
        st.image(np.array(img.resize((224, 224))), use_container_width=True)
    with g2:
        st.markdown("**Activation Heatmap**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(heat_resized, cmap='jet')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()
    with g3:
        st.markdown("**Overlay**")
        st.image(overlay, use_container_width=True)

else:
    st.markdown("Upload a waste image to begin classification and sorting.")
    st.markdown("""
    **Supported waste categories:** Cardboard, Glass, Metal, Paper, Plastic, Trash
    """)
    st.divider()
    st.markdown("**System Overview**")
    st.markdown("""
    Eco-Sort combines a deep learning classifier (MobileNetV2, 86.77% accuracy) with an 
    automated sorting framework. Each classified item is assigned to a specific conveyor lane, 
    processing action, and destination facility — simulating a real-world intelligent waste 
    management pipeline. Grad-CAM visualization provides explainability for each decision.
    """)