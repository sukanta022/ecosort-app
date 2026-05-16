import os
import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

st.set_page_config(
    page_title="Eco-Sort",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme & Global CSS ──
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Base reset ── */
* { box-sizing: border-box; margin: 0; padding: 0; }

[data-theme="dark"] {
    --bg:        #0d0f12;
    --bg2:       #141720;
    --bg3:       #1c2030;
    --border:    rgba(255,255,255,0.07);
    --text:      #e8eaf0;
    --muted:     #7a8099;
    --accent:    #3ddc84;
    --accent2:   #00b4d8;
    --danger:    #ff6b6b;
    --card:      #141720;
    --sidebar:   #0f1117;
    --tag-bg:    rgba(61,220,132,0.12);
    --tag-text:  #3ddc84;
    --shadow:    0 4px 24px rgba(0,0,0,0.5);
}
[data-theme="light"] {
    --bg:        #f4f6f9;
    --bg2:       #ffffff;
    --bg3:       #eef1f6;
    --border:    rgba(0,0,0,0.08);
    --text:      #1a1d2e;
    --muted:     #6b7280;
    --accent:    #16a34a;
    --accent2:   #0284c7;
    --danger:    #dc2626;
    --card:      #ffffff;
    --sidebar:   #ffffff;
    --tag-bg:    rgba(22,163,74,0.10);
    --tag-text:  #16a34a;
    --shadow:    0 4px 24px rgba(0,0,0,0.08);
}

/* Apply theme to Streamlit root */
.stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 240px !important;
    max-width: 240px !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Images ── */
[data-testid="stImage"] img { border-radius: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
CLASS_NAMES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
PREDICT_PATH = 'model/ecosort_model.onnx'
GRADCAM_PATH = 'model/ecosort_gradcam.onnx'

SORTING = {
    'cardboard': {'bin':'Bin A','lane':'Lane 1','action':'Flatten & Compress','dest':'Paper Recycling Facility','type':'Recyclable','color':'#92713a'},
    'glass':     {'bin':'Bin B','lane':'Lane 2','action':'Color Sort & Crush','dest':'Glass Processing Plant','type':'Recyclable','color':'#3a7abf'},
    'metal':     {'bin':'Bin C','lane':'Lane 3','action':'Magnetic Separation','dest':'Metal Scrap Facility','type':'Recyclable','color':'#5a6a7a'},
    'paper':     {'bin':'Bin D','lane':'Lane 1','action':'Stack & Bundle','dest':'Paper Recycling Facility','type':'Recyclable','color':'#5a8a5a'},
    'plastic':   {'bin':'Bin E','lane':'Lane 4','action':'Shred & Pelletize','dest':'Plastic Recycling Plant','type':'Recyclable','color':'#c4832a'},
    'trash':     {'bin':'Bin F','lane':'Lane 5','action':'Compact & Seal','dest':'Controlled Landfill','type':'Non-Recyclable','color':'#b04040'},
}

CLASS_ICONS = {
    'cardboard':'📦','glass':'🫙','metal':'🔩','paper':'📄','plastic':'🧴','trash':'🗑️'
}

# ── Session state ──
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'history' not in st.session_state:
    st.session_state.history = []

theme = st.session_state.theme

# Inject theme variable
st.markdown(f'<div id="theme-root" data-theme="{theme}" style="display:none"></div>', unsafe_allow_html=True)
st.markdown(f"""
<script>
document.documentElement.setAttribute('data-theme', '{theme}');
document.body.setAttribute('data-theme', '{theme}');
var root = document.querySelector('.stApp');
if(root) root.setAttribute('data-theme', '{theme}');
</script>
""", unsafe_allow_html=True)

# Force CSS variables via inline injection
bg        = '#0d0f12'   if theme=='dark' else '#f4f6f9'
bg2       = '#141720'   if theme=='dark' else '#ffffff'
bg3       = '#1c2030'   if theme=='dark' else '#eef1f6'
border    = 'rgba(255,255,255,0.07)' if theme=='dark' else 'rgba(0,0,0,0.08)'
text      = '#e8eaf0'   if theme=='dark' else '#1a1d2e'
muted     = '#7a8099'   if theme=='dark' else '#6b7280'
accent    = '#3ddc84'   if theme=='dark' else '#16a34a'
accent2   = '#00b4d8'   if theme=='dark' else '#0284c7'
card      = '#141720'   if theme=='dark' else '#ffffff'
sidebar   = '#0f1117'   if theme=='dark' else '#ffffff'
tag_bg    = 'rgba(61,220,132,0.12)' if theme=='dark' else 'rgba(22,163,74,0.10)'
tag_text  = '#3ddc84'   if theme=='dark' else '#16a34a'
shadow    = '0 4px 24px rgba(0,0,0,0.5)' if theme=='dark' else '0 4px 24px rgba(0,0,0,0.08)'

st.markdown(f"""
<style>
.stApp, section[data-testid="stSidebar"] {{
    --bg:{bg}; --bg2:{bg2}; --bg3:{bg3}; --border:{border};
    --text:{text}; --muted:{muted}; --accent:{accent}; --accent2:{accent2};
    --card:{card}; --sidebar:{sidebar}; --tag-bg:{tag_bg}; --tag-text:{tag_text};
    --shadow:{shadow};
    background: {bg} !important;
    color: {text} !important;
}}
section[data-testid="stSidebar"] {{ background: {sidebar} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Model loader ──
@st.cache_resource
def load_models():
    pred    = ort.InferenceSession(PREDICT_PATH)
    gradcam = ort.InferenceSession(GRADCAM_PATH)
    return pred, gradcam

def preprocess(img):
    arr = np.array(img.resize((224,224))).astype(np.float32)/255.0
    return np.expand_dims(arr,0)

def predict(session, arr):
    name  = session.get_inputs()[0].name
    preds = session.run(None,{name:arr})[0][0]
    return preds, CLASS_NAMES[np.argmax(preds)], float(np.max(preds))*100

def get_gradcam(session, arr):
    name    = session.get_inputs()[0].name
    outputs = session.run(None,{name:arr})
    conv    = outputs[0][0]
    preds   = outputs[1][0]
    top     = np.argmax(preds)
    grads   = np.zeros_like(conv)
    for i in range(conv.shape[-1]):
        grads[:,:,i] = conv[:,:,i].mean() * preds[top]
    w       = np.mean(grads,axis=(0,1))
    heatmap = np.zeros(conv.shape[:2],dtype=np.float32)
    for i,ww in enumerate(w):
        heatmap += ww*conv[:,:,i]
    heatmap = np.maximum(heatmap,0)
    if heatmap.max()>0: heatmap/=heatmap.max()
    return heatmap

def overlay(img, heatmap):
    orig = np.array(img.resize((224,224)))
    h    = cv2.resize(heatmap,(224,224))
    hc   = cv2.applyColorMap(np.uint8(255*h),cv2.COLORMAP_JET)
    hc   = cv2.cvtColor(hc,cv2.COLOR_BGR2RGB)
    ov   = cv2.addWeighted(orig,0.6,hc,0.4,0)
    return h, ov

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div style="padding:28px 20px 20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:34px;height:34px;background:{accent};border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:18px;flex-shrink:0;">♻</div>
            <div>
                <div style="font-size:17px;font-weight:600;color:{text};letter-spacing:-0.3px;">Eco-Sort</div>
                <div style="font-size:11px;color:{muted};font-family:'DM Mono',monospace;">v1.0 · Smart Waste AI</div>
            </div>
        </div>
    </div>
    <div style="height:1px;background:{border};margin:0 20px 16px;"></div>
    """, unsafe_allow_html=True)

    # Navigation
    pages = [
        ('Home',             '⬜', 'Overview & stats'),
        ('Waste Identification', '🔍', 'Classify & analyze'),
    ]

    st.markdown(f'<div style="padding:0 12px;"><div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.5px;text-transform:uppercase;padding:0 8px 8px;">Navigation</div></div>', unsafe_allow_html=True)

    for pname, icon, desc in pages:
        active = st.session_state.page == pname
        abg    = tag_bg  if active else 'transparent'
        acol   = accent  if active else muted
        if st.sidebar.button(f"{pname}", key=f"nav_{pname}", use_container_width=True):
            st.session_state.page = pname
            st.rerun()

    st.markdown(f"""
    <div style="height:1px;background:{border};margin:16px 20px;"></div>
    <div style="padding:0 20px;">
        <div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.5px;
                    text-transform:uppercase;margin-bottom:10px;">System Status</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{accent};
                        box-shadow:0 0 6px {accent};flex-shrink:0;"></div>
            <div style="font-size:12px;color:{text};">Model loaded</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{accent};
                        box-shadow:0 0 6px {accent};flex-shrink:0;"></div>
            <div style="font-size:12px;color:{text};">MobileNetV2 ready</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{accent};
                        box-shadow:0 0 6px {accent};flex-shrink:0;"></div>
            <div style="font-size:12px;color:{text};">Grad-CAM active</div>
        </div>
    </div>
    <div style="height:1px;background:{border};margin:16px 20px;"></div>
    <div style="padding:0 20px;margin-bottom:12px;">
        <div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.5px;
                    text-transform:uppercase;margin-bottom:10px;">Session Stats</div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:12px;color:{muted};">Classifications</span>
            <span style="font-size:13px;font-weight:600;color:{text};font-family:'DM Mono',monospace;">
                {len(st.session_state.history)}
            </span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:12px;color:{muted};">Accuracy</span>
            <span style="font-size:13px;font-weight:600;color:{accent};font-family:'DM Mono',monospace;">86.77%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle at bottom
    st.markdown('<div style="flex:1"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{border};margin:8px 20px 16px;"></div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([1,1])
    with col_t1:
        if st.button("☀ Light" if theme=='dark' else "☀ Light", key="light_btn", use_container_width=True):
            st.session_state.theme = 'light'
            st.rerun()
    with col_t2:
        if st.button("◑ Dark" if theme=='light' else "◑ Dark", key="dark_btn", use_container_width=True):
            st.session_state.theme = 'dark'
            st.rerun()

# ── Main content ──
main = st.container()

with main:
    # ── Topbar ──
    st.markdown(f"""
    <div style="background:{bg2};border-bottom:1px solid {border};
                padding:16px 32px;display:flex;align-items:center;
                justify-content:space-between;position:sticky;top:0;z-index:100;">
        <div>
            <div style="font-size:20px;font-weight:600;color:{text};letter-spacing:-0.4px;">
                {st.session_state.page}
            </div>
            <div style="font-size:12px;color:{muted};margin-top:1px;">
                {"Overview of your waste management AI system" if st.session_state.page=="Home" else "Upload and classify waste images with AI"}
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:{tag_bg};color:{tag_text};font-size:11px;font-weight:500;
                        padding:4px 12px;border-radius:20px;font-family:'DM Mono',monospace;">
                MobileNetV2 · 86.77%
            </div>
        </div>
    </div>
    <div style="padding:28px 32px;">
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    #               HOME PAGE
    # ════════════════════════════════════════
    if st.session_state.page == 'Home':

        # Stat cards
        stats = [
            ('86.77%',  'Test Accuracy',       accent,  'MobileNetV2 Phase 1'),
            ('6',       'Waste Classes',        accent2, 'Cardboard to Trash'),
            ('431',     'Test Samples',         '#a78bfa','Evaluation dataset'),
            ('0.85',    'F1-Score',             '#fb923c','Weighted average'),
        ]
        cols = st.columns(4)
        for i,(val,label,color,sub) in enumerate(stats):
            with cols[i]:
                st.markdown(f"""
                <div style="background:{card};border:1px solid {border};border-radius:16px;
                            padding:20px;position:relative;overflow:hidden;">
                    <div style="position:absolute;top:0;left:0;right:0;height:3px;
                                background:{color};border-radius:16px 16px 0 0;"></div>
                    <div style="font-size:28px;font-weight:600;color:{text};
                                font-family:'DM Mono',monospace;letter-spacing:-1px;
                                margin-bottom:4px;">{val}</div>
                    <div style="font-size:13px;font-weight:500;color:{text};margin-bottom:2px;">{label}</div>
                    <div style="font-size:11px;color:{muted};">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # Class performance grid
        col_a, col_b = st.columns([3,2])

        with col_a:
            st.markdown(f"""
            <div style="background:{card};border:1px solid {border};border-radius:16px;padding:22px;">
                <div style="font-size:15px;font-weight:600;color:{text};margin-bottom:16px;">
                    Class Performance
                </div>
            """, unsafe_allow_html=True)

            perf = [
                ('cardboard', 0.93, '#92713a'),
                ('paper',     0.91, '#5a8a5a'),
                ('glass',     0.83, '#3a7abf'),
                ('metal',     0.82, '#5a6a7a'),
                ('plastic',   0.81, '#c4832a'),
                ('trash',     0.69, '#b04040'),
            ]
            for cls, score, color in perf:
                pct = int(score*100)
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                        <span style="font-size:13px;color:{text};text-transform:capitalize;">{cls}</span>
                        <span style="font-size:12px;color:{muted};font-family:'DM Mono',monospace;">F1: {score}</span>
                    </div>
                    <div style="background:{bg3};border-radius:6px;height:7px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:{color};
                                    border-radius:6px;transition:width 0.8s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="background:{card};border:1px solid {border};border-radius:16px;padding:22px;height:100%;">
                <div style="font-size:15px;font-weight:600;color:{text};margin-bottom:16px;">
                    Sorting Lanes
                </div>
            """, unsafe_allow_html=True)

            lanes = [
                ('Lane 1','Cardboard, Paper','#5a8a5a'),
                ('Lane 2','Glass','#3a7abf'),
                ('Lane 3','Metal','#5a6a7a'),
                ('Lane 4','Plastic','#c4832a'),
                ('Lane 5','Trash','#b04040'),
            ]
            for lane, items, color in lanes:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;
                            background:{bg3};border-radius:10px;padding:10px 14px;">
                    <div style="width:8px;height:8px;border-radius:50%;
                                background:{color};flex-shrink:0;"></div>
                    <div>
                        <div style="font-size:12px;font-weight:500;color:{text};">{lane}</div>
                        <div style="font-size:11px;color:{muted};">{items}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # Recent history
        st.markdown(f"""
        <div style="background:{card};border:1px solid {border};border-radius:16px;padding:22px;">
            <div style="font-size:15px;font-weight:600;color:{text};margin-bottom:16px;">
                Recent Classifications
            </div>
        """, unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown(f"""
            <div style="text-align:center;padding:32px;color:{muted};">
                <div style="font-size:32px;margin-bottom:8px;">🔍</div>
                <div style="font-size:14px;">No classifications yet — go to Waste Identification to start</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;
                        font-size:11px;color:{muted};font-weight:600;
                        letter-spacing:0.8px;text-transform:uppercase;
                        padding:0 0 10px;border-bottom:1px solid {border};margin-bottom:8px;">
                <span>#</span><span>Class</span><span>Confidence</span>
                <span>Bin</span><span>Recyclable</span>
            </div>
            """, unsafe_allow_html=True)
            for i, h in enumerate(reversed(st.session_state.history[-8:])):
                s = SORTING[h['cls']]
                rec_color = accent if s['type']=='Recyclable' else '#b04040'
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;
                            font-size:13px;color:{text};padding:8px 0;
                            border-bottom:1px solid {border};">
                    <span style="color:{muted};font-family:'DM Mono',monospace;">{len(st.session_state.history)-i:02d}</span>
                    <span style="text-transform:capitalize;font-weight:500;">{h['cls']}</span>
                    <span style="font-family:'DM Mono',monospace;color:{accent};">{h['conf']:.1f}%</span>
                    <span style="color:{muted};">{s['bin']}</span>
                    <span style="color:{rec_color};font-size:11px;font-weight:500;">{s['type']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Quick access button
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        if st.button("Go to Waste Identification →", use_container_width=False):
            st.session_state.page = 'Waste Identification'
            st.rerun()

    # ════════════════════════════════════════
    #         WASTE IDENTIFICATION PAGE
    # ════════════════════════════════════════
    elif st.session_state.page == 'Waste Identification':

        try:
            pred_sess, gcam_sess = load_models()
            models_ready = True
        except Exception as e:
            models_ready = False
            st.markdown(f"""
            <div style="background:rgba(176,64,64,0.12);border:1px solid #b04040;border-radius:12px;
                        padding:16px 20px;color:#ff9999;font-size:13px;margin-bottom:20px;">
                Model files not found. Please place <code>ecosort_model.onnx</code> and
                <code>ecosort_gradcam.onnx</code> in the <code>model/</code> folder.
            </div>
            """, unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            # Upload section
            st.markdown(f"""
            <div style="background:{card};border:1px solid {border};border-radius:16px;
                        padding:22px;margin-bottom:20px;">
                <div style="font-size:15px;font-weight:600;color:{text};margin-bottom:4px;">
                    Upload Image
                </div>
                <div style="font-size:12px;color:{muted};margin-bottom:16px;">
                    Supports JPG, JPEG, PNG
                </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader("", type=['jpg','jpeg','png'], label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            if uploaded:
                img = Image.open(uploaded).convert('RGB')
                st.markdown(f"""
                <div style="background:{card};border:1px solid {border};border-radius:16px;
                            padding:22px;">
                    <div style="font-size:13px;font-weight:500;color:{muted};margin-bottom:12px;">
                        Preview
                    </div>
                """, unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;margin-top:12px;">
                        <span style="font-size:11px;color:{muted};">Size</span>
                        <span style="font-size:11px;color:{text};font-family:'DM Mono',monospace;">
                            {img.width} × {img.height} px
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            if uploaded and models_ready:
                arr = preprocess(img)

                with st.spinner("Analyzing..."):
                    probs, pred_cls, conf = predict(pred_sess, arr)
                    heatmap              = get_gradcam(gcam_sess, arr)
                    heat_res, ov         = overlay(img, heatmap)
                    time.sleep(0.3)

                # Save to history
                st.session_state.history.append({'cls':pred_cls,'conf':conf})

                sort = SORTING[pred_cls]
                icon = CLASS_ICONS.get(pred_cls,'♻️')
                rec_color = accent if sort['type']=='Recyclable' else '#b04040'

                # ── Result card ──
                st.markdown(f"""
                <div style="background:{card};border:1px solid {border};border-radius:16px;
                            padding:22px;margin-bottom:20px;">
                    <div style="font-size:11px;font-weight:600;color:{muted};letter-spacing:1.2px;
                                text-transform:uppercase;margin-bottom:14px;">Classification Result</div>
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">
                        <div style="width:52px;height:52px;border-radius:14px;
                                    background:{bg3};display:flex;align-items:center;
                                    justify-content:center;font-size:26px;flex-shrink:0;">{icon}</div>
                        <div>
                            <div style="font-size:26px;font-weight:600;color:{text};
                                        letter-spacing:-0.5px;text-transform:capitalize;">{pred_cls}</div>
                            <div style="font-size:12px;color:{muted};margin-top:2px;">
                                Detected material type
                            </div>
                        </div>
                        <div style="margin-left:auto;text-align:right;">
                            <div style="font-size:28px;font-weight:600;color:{accent};
                                        font-family:'DM Mono',monospace;">{conf:.1f}%</div>
                            <div style="font-size:11px;color:{muted};">confidence</div>
                        </div>
                    </div>
                    <div style="background:{bg3};border-radius:10px;height:6px;overflow:hidden;margin-bottom:18px;">
                        <div style="width:{conf:.1f}%;height:100%;background:{accent};border-radius:10px;"></div>
                    </div>
                    <div style="font-size:12px;font-weight:600;color:{muted};letter-spacing:0.8px;
                                text-transform:uppercase;margin-bottom:10px;">All classes</div>
                """, unsafe_allow_html=True)

                for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x:x[1], reverse=True):
                    is_top = cls == pred_cls
                    bar_bg = accent if is_top else accent2
                    bar_w  = f"{prob*100:.1f}"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                        <span style="width:76px;font-size:12px;color:{'#fff' if is_top else muted};
                                     font-weight:{'500' if is_top else '400'};
                                     text-transform:capitalize;">{cls}</span>
                        <div style="flex:1;background:{bg3};border-radius:5px;height:5px;overflow:hidden;">
                            <div style="width:{bar_w}%;height:100%;background:{bar_bg};border-radius:5px;"></div>
                        </div>
                        <span style="width:40px;font-size:11px;color:{muted};text-align:right;
                                     font-family:'DM Mono',monospace;">{prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Sorting decision ──
                st.markdown(f"""
                <div style="background:{card};border:1px solid {border};border-radius:16px;
                            padding:22px;margin-bottom:20px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
                        <div style="font-size:11px;font-weight:600;color:{muted};letter-spacing:1.2px;
                                    text-transform:uppercase;">Sorting Decision</div>
                        <div style="background:{'rgba(61,220,132,0.12)' if sort['type']=='Recyclable' else 'rgba(176,64,64,0.12)'};
                                    color:{rec_color};font-size:11px;font-weight:500;
                                    padding:3px 10px;border-radius:20px;">{sort['type']}</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                        <div style="background:{bg3};border-radius:12px;padding:14px;">
                            <div style="font-size:10px;color:{muted};text-transform:uppercase;
                                        letter-spacing:0.8px;margin-bottom:5px;">Assigned Bin</div>
                            <div style="font-size:15px;font-weight:600;color:{text};">{sort['bin']}</div>
                        </div>
                        <div style="background:{bg3};border-radius:12px;padding:14px;">
                            <div style="font-size:10px;color:{muted};text-transform:uppercase;
                                        letter-spacing:0.8px;margin-bottom:5px;">Conveyor Lane</div>
                            <div style="font-size:15px;font-weight:600;color:{text};">{sort['lane']}</div>
                        </div>
                        <div style="background:{bg3};border-radius:12px;padding:14px;">
                            <div style="font-size:10px;color:{muted};text-transform:uppercase;
                                        letter-spacing:0.8px;margin-bottom:5px;">Processing Action</div>
                            <div style="font-size:13px;font-weight:500;color:{text};">{sort['action']}</div>
                        </div>
                        <div style="background:{bg3};border-radius:12px;padding:14px;">
                            <div style="font-size:10px;color:{muted};text-transform:uppercase;
                                        letter-spacing:0.8px;margin-bottom:5px;">Destination</div>
                            <div style="font-size:12px;font-weight:500;color:{text};">{sort['dest']}</div>
                        </div>
                    </div>
                    <div style="margin-top:14px;">
                        <div style="font-size:10px;color:{muted};text-transform:uppercase;
                                    letter-spacing:0.8px;margin-bottom:8px;">Conveyor Pipeline</div>
                        <div style="display:flex;gap:6px;">
                """, unsafe_allow_html=True)

                all_lanes = ['Lane 1','Lane 2','Lane 3','Lane 4','Lane 5']
                for ln in all_lanes:
                    is_active = sort['lane'] == ln
                    st.markdown(f"""
                    <div style="flex:1;padding:7px 4px;border-radius:8px;text-align:center;
                                font-size:11px;font-weight:{'600' if is_active else '400'};
                                background:{''+accent+'' if is_active else bg3};
                                color:{'#000' if is_active else muted};
                                transition:all 0.2s;">{ln}</div>
                    """, unsafe_allow_html=True)

                st.markdown("</div></div></div>", unsafe_allow_html=True)

                # ── Grad-CAM ──
                st.markdown(f"""
                <div style="background:{card};border:1px solid {border};border-radius:16px;padding:22px;">
                    <div style="font-size:11px;font-weight:600;color:{muted};letter-spacing:1.2px;
                                text-transform:uppercase;margin-bottom:4px;">Grad-CAM Explainability</div>
                    <div style="font-size:12px;color:{muted};margin-bottom:14px;">
                        Regions the model focused on to make its decision
                    </div>
                """, unsafe_allow_html=True)

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.markdown(f'<div style="font-size:11px;color:{muted};margin-bottom:6px;">Original</div>', unsafe_allow_html=True)
                    st.image(np.array(img.resize((224,224))), use_container_width=True)
                with g2:
                    st.markdown(f'<div style="font-size:11px;color:{muted};margin-bottom:6px;">Activation map</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(3,3))
                    fig.patch.set_facecolor('none')
                    ax.imshow(heat_res, cmap='jet')
                    ax.axis('off')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
                    buf.seek(0)
                    st.image(buf, use_container_width=True)
                    plt.close()
                with g3:
                    st.markdown(f'<div style="font-size:11px;color:{muted};margin-bottom:6px;">Overlay</div>', unsafe_allow_html=True)
                    st.image(ov, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

            elif not uploaded:
                # Empty state
                st.markdown(f"""
                <div style="background:{card};border:1px solid {border};border-radius:16px;
                            padding:60px 32px;text-align:center;height:100%;">
                    <div style="font-size:48px;margin-bottom:16px;opacity:0.4;">🔍</div>
                    <div style="font-size:16px;font-weight:500;color:{text};margin-bottom:8px;">
                        No image uploaded yet
                    </div>
                    <div style="font-size:13px;color:{muted};max-width:280px;margin:0 auto;line-height:1.6;">
                        Upload a waste image on the left to classify it and get a sorting decision
                    </div>
                    <div style="margin-top:24px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
                        {''.join([f'<span style="background:{bg3};color:{muted};font-size:11px;padding:4px 12px;border-radius:20px;">{c}</span>' for c in CLASS_NAMES])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)