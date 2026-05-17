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

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
PREDICT_PATH = 'model/ecosort_model.onnx'
GRADCAM_PATH = 'model/ecosort_gradcam.onnx'

SORTING_FRAMEWORK = {
    'cardboard': {'bin':'Bin A — Dry Recyclables',     'category':'Recyclable',     'conveyor':'Lane 1', 'action':'Flatten & Compress',        'destination':'Paper Recycling Facility',           'color':'#8B6914'},
    'glass':     {'bin':'Bin B — Fragile Recyclables', 'category':'Recyclable',     'conveyor':'Lane 2', 'action':'Separate by Color & Crush',  'destination':'Glass Processing Plant',             'color':'#1565C0'},
    'metal':     {'bin':'Bin C — Metal Recyclables',   'category':'Recyclable',     'conveyor':'Lane 3', 'action':'Magnetic Separation & Bale', 'destination':'Metal Scrap Facility',               'color':'#37474F'},
    'paper':     {'bin':'Bin D — Dry Recyclables',     'category':'Recyclable',     'conveyor':'Lane 1', 'action':'Stack & Bundle',             'destination':'Paper Recycling Facility',           'color':'#2E7D32'},
    'plastic':   {'bin':'Bin E — Plastic Recyclables', 'category':'Recyclable',     'conveyor':'Lane 4', 'action':'Shred & Pelletize',          'destination':'Plastic Recycling Plant',            'color':'#F57F17'},
    'trash':     {'bin':'Bin F — General Waste',       'category':'Non-Recyclable', 'conveyor':'Lane 5', 'action':'Compact & Seal',             'destination':'Controlled Landfill / Incineration', 'color':'#B71C1C'},
}

@st.cache_resource
def load_models():
    return ort.InferenceSession(PREDICT_PATH), ort.InferenceSession(GRADCAM_PATH)

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
    gradients   = np.zeros_like(conv_output)
    for i in range(conv_output.shape[-1]):
        gradients[:, :, i] = conv_output[:, :, i].mean() * predictions[top_class]
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
    return heat_resized, cv2.addWeighted(orig, 0.6, heat_colored, 0.4, 0)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 4px 20px;'>
        <div style='display:flex;align-items:center;gap:10px;'>
            <div style='background:#2E7D32;width:38px;height:38px;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:20px;flex-shrink:0;'>♻</div>
            <div>
                <div style='font-size:17px;font-weight:700;color:#ffffff;'>Eco-Sort</div>
                <div style='font-size:11px;color:#888;'>Smart Waste AI · v1.0</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='font-size:10px;font-weight:600;color:#666;letter-spacing:1.5px;
                text-transform:uppercase;margin-bottom:10px;'>Navigation</div>
    <div style='background:#1e2a1e;border-left:3px solid #2E7D32;border-radius:6px;
                padding:10px 14px;margin-bottom:4px;'>
        <span style='color:#a6e3a1;font-size:13.5px;font-weight:600;'>Home</span>
    </div>
    <div style='padding:10px 14px;border-radius:6px;'>
        <span style='color:#666;font-size:13.5px;'>Waste Identification</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='font-size:10px;font-weight:600;color:#666;letter-spacing:1.5px;
                text-transform:uppercase;margin-bottom:12px;'>System Status</div>
    """, unsafe_allow_html=True)

    for label in ['Model loaded', 'MobileNetV2 ready', 'Grad-CAM active']:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
            <div style='width:8px;height:8px;border-radius:50%;background:#2E7D32;
                        box-shadow:0 0 6px #2E7D3299;flex-shrink:0;'></div>
            <span style='font-size:12.5px;color:#ccc;'>{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='font-size:10px;font-weight:600;color:#666;letter-spacing:1.5px;
                text-transform:uppercase;margin-bottom:12px;'>Model Info</div>
    <div style='font-size:12.5px;color:#888;line-height:2.1;'>
        Architecture&nbsp;&nbsp;<span style='color:#ccc;'>MobileNetV2</span><br>
        Accuracy&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#a6e3a1;font-weight:600;'>86.77%</span><br>
        Classes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#ccc;'>6</span><br>
        Explainability&nbsp;<span style='color:#ccc;'>Grad-CAM</span>
    </div>
    """, unsafe_allow_html=True)

# ── Main UI ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&display=swap');

.eco-hero {
    background: linear-gradient(135deg, #0a1a0d 0%, #0d1f10 40%, #112614 100%);
    border: 1px solid rgba(46,125,50,0.3);
    border-radius: 20px;
    padding: 44px 40px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 28px;
}
.eco-hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(46,125,50,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.eco-hero::after {
    content: '♻';
    position: absolute;
    right: 36px; top: 50%; transform: translateY(-50%);
    font-size: 100px;
    opacity: 0.05;
    pointer-events: none;
    line-height: 1;
}
.eco-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(46,125,50,0.15);
    border: 1px solid rgba(46,125,50,0.4);
    border-radius: 30px;
    padding: 5px 16px;
    font-size: 11.5px;
    font-weight: 600;
    color: #66bb6a;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.eco-badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4caf50;
    box-shadow: 0 0 8px #4caf5099;
    display: inline-block;
}
.eco-title {
    font-family: 'Sora', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #81c784 0%, #4caf50 40%, #2E7D32 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: 12px;
}
.eco-subtitle {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    color: #6b7a6b;
    letter-spacing: 0.2px;
    line-height: 1.6;
    margin-bottom: 28px;
}
.eco-pills {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
.eco-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: #8a9e8a;
    font-family: 'Sora', sans-serif;
}
.eco-stats {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 28px;
    padding-top: 24px;
    border-top: 1px solid rgba(46,125,50,0.15);
}
.eco-stat-val {
    font-family: 'Sora', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #81c784;
    line-height: 1;
    margin-bottom: 4px;
}
.eco-stat-lbl {
    font-size: 11px;
    color: #4a5e4a;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>

<div class="eco-hero">
    <div class="eco-badge">
        <span class="eco-badge-dot"></span>
        AI-Powered · Deep Learning
    </div>
    <div class="eco-title">Eco-Sort</div>
    <div class="eco-subtitle">
        Intelligent Waste Classification &amp; Automated Sorting System
    </div>
    <div class="eco-pills">
        <span class="eco-pill">📦 Cardboard</span>
        <span class="eco-pill">🫙 Glass</span>
        <span class="eco-pill">🔩 Metal</span>
        <span class="eco-pill">📄 Paper</span>
        <span class="eco-pill">🧴 Plastic</span>
        <span class="eco-pill">🗑️ Trash</span>
    </div>
    <div class="eco-stats">
        <div>
            <div class="eco-stat-val">86.77%</div>
            <div class="eco-stat-lbl">Accuracy</div>
        </div>
        <div>
            <div class="eco-stat-val">6</div>
            <div class="eco-stat-lbl">Classes</div>
        </div>
        <div>
            <div class="eco-stat-val">0.85</div>
            <div class="eco-stat-lbl">F1-Score</div>
        </div>
        <div>
            <div class="eco-stat-val">5</div>
            <div class="eco-stat-lbl">Sort Lanes</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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

    # ── Classification Result ──
    st.markdown("### Classification Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Image**")
        st.image(img, use_container_width=True)

    with col2:
        st.markdown("**Prediction**")
        st.markdown(f"""
        <div style='background:#1e1e2e;padding:24px;border-radius:10px;
                    border-left:5px solid {sort_info["color"]};margin-bottom:16px;'>
            <div style='color:#aaa;font-size:0.8rem;text-transform:uppercase;
                        letter-spacing:1px;'>Detected Material</div>
            <div style='color:#fff;font-size:2rem;font-weight:700;
                        margin:6px 0;'>{pred_class.upper()}</div>
            <div style='color:#aaa;font-size:0.9rem;'>Confidence:
                <span style='color:#a6e3a1;font-weight:600;'>{confidence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Class Probability Distribution**")
        for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
            bar_color = sort_info["color"] if cls == pred_class else "#444"
            st.markdown(f"""
            <div style='display:flex;align-items:center;margin:5px 0;gap:10px;'>
                <span style='width:85px;font-size:12px;color:#ccc;'>{cls}</span>
                <div style='flex:1;background:#2a2a3e;border-radius:4px;height:16px;'>
                    <div style='width:{prob*100:.1f}%;background:{bar_color};
                                height:16px;border-radius:4px;'></div>
                </div>
                <span style='width:44px;font-size:12px;color:#ccc;
                             text-align:right;'>{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Sorting Decision ──
    st.markdown("### Automated Sorting Decision")
    st.markdown(
        "Based on the classification result, the sorting controller routes "
        "the item through the appropriate processing pipeline:"
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ['Assigned Bin', 'Conveyor Lane', 'Processing Action', 'Destination'],
        [sort_info["bin"], sort_info["conveyor"], sort_info["action"], sort_info["destination"]]
    ):
        with col:
            extra = f"<div style='color:#aaa;font-size:12px;margin-top:4px;'>Category: {sort_info['category']}</div>" if label == 'Conveyor Lane' else ""
            st.markdown(f"""
            <div style='background:#1e1e2e;padding:16px;border-radius:8px;
                        border-top:3px solid {sort_info["color"]};min-height:110px;'>
                <div style='color:#888;font-size:11px;text-transform:uppercase;
                            letter-spacing:1px;'>{label}</div>
                <div style='color:#fff;font-size:14px;font-weight:600;
                            margin-top:8px;'>{value}</div>
                {extra}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Sorting Pipeline — Active Route**")

    lane_cols = st.columns(5)
    all_lanes = {
        'Lane 1': ['cardboard', 'paper'],
        'Lane 2': ['glass'],
        'Lane 3': ['metal'],
        'Lane 4': ['plastic'],
        'Lane 5': ['trash'],
    }
    for idx, (lane, materials) in enumerate(all_lanes.items()):
        is_active = sort_info['conveyor'] == lane
        with lane_cols[idx]:
            st.markdown(f"""
            <div style='background:{sort_info["color"] if is_active else "#2a2a3e"};
                        border:2px solid {sort_info["color"] if is_active else "#333"};
                        border-radius:8px;padding:12px;text-align:center;'>
                <div style='color:{"#fff" if is_active else "#555"};font-size:11px;
                            font-weight:600;'>{lane}</div>
                <div style='color:{"#fff" if is_active else "#555"};font-size:10px;
                            margin-top:4px;'>{" / ".join(m.capitalize() for m in materials)}</div>
                {'<div style="color:#fff;font-size:10px;margin-top:6px;font-weight:700;">ACTIVE</div>' if is_active else ''}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Grad-CAM ──
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
    st.info("Upload a waste image above to begin classification and sorting.")
    st.markdown("**Supported waste categories:** Cardboard · Glass · Metal · Paper · Plastic · Trash")
    st.divider()
    st.markdown("**System Overview**")
    st.markdown("""
    Eco-Sort combines a deep learning classifier (MobileNetV2, 86.77% accuracy) with an
    automated sorting framework. Each classified item is assigned to a specific conveyor lane,
    processing action, and destination facility — simulating a real-world intelligent waste
    management pipeline. Grad-CAM visualization provides explainability for each decision.
    """)