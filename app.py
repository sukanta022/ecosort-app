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

# ── Session state ──
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "history" not in st.session_state:
    st.session_state.history = []

T = st.session_state.theme

# ── Color palette ──
if T == "dark":
    BG      = "#0d0f12"
    BG2     = "#141720"
    BG3     = "#1c2030"
    BORDER  = "rgba(255,255,255,0.08)"
    TEXT    = "#e8eaf0"
    MUTED   = "#7a8099"
    ACCENT  = "#3ddc84"
    ACCENT2 = "#00b4d8"
    CARD    = "#141720"
    SIDEBAR = "#0f1117"
    TAGBG   = "rgba(61,220,132,0.12)"
    TAGTXT  = "#3ddc84"
else:
    BG      = "#f0f2f6"
    BG2     = "#ffffff"
    BG3     = "#e8ecf2"
    BORDER  = "rgba(0,0,0,0.09)"
    TEXT    = "#1a1d2e"
    MUTED   = "#6b7280"
    ACCENT  = "#16a34a"
    ACCENT2 = "#0284c7"
    CARD    = "#ffffff"
    SIDEBAR = "#ffffff"
    TAGBG   = "rgba(22,163,74,0.10)"
    TAGTXT  = "#16a34a"

# ── ONE single CSS injection ──
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  .stApp {{ background: {BG} !important; font-family: 'DM Sans', sans-serif !important; color: {TEXT} !important; }}
  #MainMenu, footer, header {{ visibility: hidden !important; display: none !important; }}
  .block-container {{ padding: 0 !important; max-width: 100% !important; }}
  section[data-testid="stSidebar"] > div:first-child {{ padding: 0 !important; }}
  section[data-testid="stSidebar"] {{
    background: {SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
    min-width: 240px !important;
    max-width: 240px !important;
  }}
  section[data-testid="stSidebar"] * {{ font-family: 'DM Sans', sans-serif !important; color: {TEXT} !important; }}
  .stButton > button {{
    background: {ACCENT} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 9px 20px !important;
    transition: opacity .2s !important;
    width: 100%;
  }}
  .stButton > button:hover {{ opacity: .82 !important; }}
  .stButton > button *, .stButton > button p {{ color: #ffffff !important; }}
  [data-testid="stFileUploader"] {{
    background: {BG3} !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 14px !important;
  }}
  [data-testid="stFileUploaderDropzone"] {{ background: transparent !important; }}
  [data-testid="stFileUploader"] * {{ color: {TEXT} !important; }}
  .stSpinner > div {{ border-top-color: {ACCENT} !important; }}
  [data-testid="stImage"] img {{ border-radius: 10px !important; }}
  ::-webkit-scrollbar {{ width: 4px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 10px; }}
  .viewerBadge_container__1QSob {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ──
CLASS_NAMES  = ["cardboard","glass","metal","paper","plastic","trash"]
PREDICT_PATH = "model/ecosort_model.onnx"
GRADCAM_PATH = "model/ecosort_gradcam.onnx"

SORTING = {
    "cardboard": {"bin":"Bin A","lane":"Lane 1","action":"Flatten & Compress",  "dest":"Paper Recycling Facility","type":"Recyclable",    "color":"#a07840"},
    "glass":     {"bin":"Bin B","lane":"Lane 2","action":"Color Sort & Crush",  "dest":"Glass Processing Plant",  "type":"Recyclable",    "color":"#3a7abf"},
    "metal":     {"bin":"Bin C","lane":"Lane 3","action":"Magnetic Separation", "dest":"Metal Scrap Facility",    "type":"Recyclable",    "color":"#5a6a7a"},
    "paper":     {"bin":"Bin D","lane":"Lane 1","action":"Stack & Bundle",      "dest":"Paper Recycling Facility","type":"Recyclable",    "color":"#5a8a5a"},
    "plastic":   {"bin":"Bin E","lane":"Lane 4","action":"Shred & Pelletize",   "dest":"Plastic Recycling Plant", "type":"Recyclable",    "color":"#c4832a"},
    "trash":     {"bin":"Bin F","lane":"Lane 5","action":"Compact & Seal",      "dest":"Controlled Landfill",     "type":"Non-Recyclable","color":"#b04040"},
}

ICONS = {"cardboard":"📦","glass":"🫙","metal":"🔩","paper":"📄","plastic":"🧴","trash":"🗑️"}

LANES = [
    ("Lane 1","Cardboard / Paper","#5a8a5a"),
    ("Lane 2","Glass",            "#3a7abf"),
    ("Lane 3","Metal",            "#5a6a7a"),
    ("Lane 4","Plastic",          "#c4832a"),
    ("Lane 5","Trash",            "#b04040"),
]

# ── Small helpers ──
def slabel(txt):
    return (f'<div style="font-size:10px;font-weight:700;color:{MUTED};letter-spacing:1.4px;'
            f'text-transform:uppercase;margin-bottom:12px;">{txt}</div>')

# ── Model ──
@st.cache_resource
def load_models():
    return ort.InferenceSession(PREDICT_PATH), ort.InferenceSession(GRADCAM_PATH)

def preprocess(img):
    arr = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(sess, arr):
    name  = sess.get_inputs()[0].name
    probs = sess.run(None, {name: arr})[0][0]
    idx   = int(np.argmax(probs))
    return probs, CLASS_NAMES[idx], float(probs[idx]) * 100

def get_gradcam(sess, arr):
    name  = sess.get_inputs()[0].name
    out   = sess.run(None, {name: arr})
    conv, preds = out[0][0], out[1][0]
    w     = np.mean(conv * preds[np.argmax(preds)], axis=(0,1))
    hm    = np.maximum(np.sum(conv * w, axis=-1), 0)
    if hm.max() > 0: hm /= hm.max()
    return hm

def make_overlay(img, hm):
    orig = np.array(img.resize((224,224)))
    h    = cv2.resize(hm, (224,224))
    hc   = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*h), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return h, cv2.addWeighted(orig, 0.6, hc, 0.4, 0)

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:24px 20px 16px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:36px;height:36px;background:{ACCENT};border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:18px;color:#fff;flex-shrink:0;">♻</div>
        <div>
          <div style="font-size:16px;font-weight:700;color:{TEXT};">Eco-Sort</div>
          <div style="font-size:11px;color:{MUTED};font-family:'DM Mono',monospace;">v1.0 · Smart Waste AI</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:{BORDER};margin:0 16px 16px;"></div>
    <div style="padding:0 12px 6px;">
      <div style="font-size:10px;font-weight:700;color:{MUTED};letter-spacing:1.4px;text-transform:uppercase;padding:0 8px 8px;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    for pname in ["Home", "Waste Identification"]:
        if st.button(pname, key=f"nav_{pname}", use_container_width=True):
            st.session_state.page = pname
            st.rerun()

    status_dots = "".join([
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        f'<div style="width:7px;height:7px;border-radius:50%;background:{ACCENT};box-shadow:0 0 6px {ACCENT};flex-shrink:0;"></div>'
        f'<div style="font-size:12px;color:{TEXT};">{s}</div></div>'
        for s in ["Model loaded", "MobileNetV2 ready", "Grad-CAM active"]
    ])

    st.markdown(f"""
    <div style="height:1px;background:{BORDER};margin:16px 16px;"></div>
    <div style="padding:0 20px;">
      <div style="font-size:10px;font-weight:700;color:{MUTED};letter-spacing:1.4px;text-transform:uppercase;margin-bottom:12px;">System Status</div>
      {status_dots}
    </div>
    <div style="height:1px;background:{BORDER};margin:16px 16px;"></div>
    <div style="padding:0 20px 16px;">
      <div style="font-size:10px;font-weight:700;color:{MUTED};letter-spacing:1.4px;text-transform:uppercase;margin-bottom:10px;">Session Stats</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:12px;color:{MUTED};">Classifications</span>
        <span style="font-size:13px;font-weight:700;color:{TEXT};font-family:'DM Mono',monospace;">{len(st.session_state.history)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;color:{MUTED};">Accuracy</span>
        <span style="font-size:13px;font-weight:700;color:{ACCENT};font-family:'DM Mono',monospace;">86.77%</span>
      </div>
    </div>
    <div style="height:1px;background:{BORDER};margin:0 16px 14px;"></div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀ Light", key="btn_light", use_container_width=True):
            st.session_state.theme = "light"; st.rerun()
    with c2:
        if st.button("◑ Dark", key="btn_dark", use_container_width=True):
            st.session_state.theme = "dark";  st.rerun()

# ════════════════════════════════════════
# TOP BAR
# ════════════════════════════════════════
page_sub = {
    "Home": "Overview of your waste management AI system",
    "Waste Identification": "Upload and classify waste images with AI",
}
st.markdown(f"""
<div style="background:{BG2};border-bottom:1px solid {BORDER};padding:16px 32px;
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-size:20px;font-weight:700;color:{TEXT};letter-spacing:-0.4px;">{st.session_state.page}</div>
    <div style="font-size:12px;color:{MUTED};margin-top:2px;">{page_sub[st.session_state.page]}</div>
  </div>
  <div style="background:{TAGBG};color:{TAGTXT};font-size:11px;font-weight:600;
              padding:5px 14px;border-radius:20px;font-family:'DM Mono',monospace;">
    MobileNetV2 · 86.77%
  </div>
</div>
<div style="padding:24px 28px 40px;">
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════
if st.session_state.page == "Home":

    # Overview banner
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                padding:24px 26px;margin-bottom:22px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;
                  background:linear-gradient(90deg,{ACCENT},{ACCENT2});"></div>
      <div style="display:flex;align-items:flex-start;gap:16px;">
        <div style="font-size:30px;margin-top:2px;flex-shrink:0;">♻️</div>
        <div>
          <div style="font-size:16px;font-weight:700;color:{TEXT};margin-bottom:6px;">
            Eco-Sort — Automated Waste Classification &amp; Intelligent Sorting System
          </div>
          <div style="font-size:11px;color:{MUTED};margin-bottom:10px;font-family:'DM Mono',monospace;">
            MobileNetV2 Transfer Learning &nbsp;·&nbsp; Grad-CAM Explainability &nbsp;·&nbsp; Rule-based Sorting Controller
          </div>
          <div style="font-size:13px;color:{MUTED};line-height:1.75;max-width:820px;">
            Eco-Sort classifies waste images into six categories —
            <strong style="color:{TEXT};">Cardboard, Glass, Metal, Paper, Plastic,</strong> and
            <strong style="color:{TEXT};">Trash</strong> — using MobileNetV2 pre-trained on ImageNet.
            Each prediction includes a <strong style="color:{TEXT};">Grad-CAM heatmap</strong> for visual explainability
            and an <strong style="color:{TEXT};">automated sorting decision</strong> that maps the item to a designated
            bin, conveyor lane, processing action, and destination facility.
            The Phase 1 model achieves <strong style="color:{ACCENT};">86.77% test accuracy</strong> on 431 held-out samples,
            outperforming the fine-tuned Phase 2 model (84.22%) which showed slight overfitting.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    stats = [
        ("86.77%","Test Accuracy",  ACCENT,   "MobileNetV2 Phase 1"),
        ("6",     "Waste Classes",  ACCENT2,  "Cardboard to Trash"),
        ("431",   "Test Samples",   "#a78bfa","Evaluation dataset"),
        ("0.85",  "F1-Score",       "#fb923c","Weighted average"),
    ]
    cols = st.columns(4)
    for i, (val, label, color, sub) in enumerate(stats):
        with cols[i]:
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                        padding:20px;position:relative;overflow:hidden;margin-bottom:20px;">
              <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{color};"></div>
              <div style="font-size:28px;font-weight:700;color:{TEXT};font-family:'DM Mono',monospace;
                          letter-spacing:-1px;margin-bottom:4px;">{val}</div>
              <div style="font-size:13px;font-weight:600;color:{TEXT};margin-bottom:3px;">{label}</div>
              <div style="font-size:11px;color:{MUTED};">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # Performance + Lanes
    col_a, col_b = st.columns([3,2], gap="medium")

    with col_a:
        perf = [
            ("cardboard",0.93,"#a07840"),("paper",0.91,"#5a8a5a"),
            ("glass",0.83,"#3a7abf"),("metal",0.82,"#5a6a7a"),
            ("plastic",0.81,"#c4832a"),("trash",0.69,"#b04040"),
        ]
        rows = ""
        for cls, score, color in perf:
            rows += f"""
            <div style="margin-bottom:13px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-size:13px;color:{TEXT};text-transform:capitalize;">{cls}</span>
                <span style="font-size:12px;color:{MUTED};font-family:'DM Mono',monospace;">F1: {score:.2f}</span>
              </div>
              <div style="background:{BG3};border-radius:6px;height:7px;overflow:hidden;">
                <div style="width:{int(score*100)}%;height:100%;background:{color};border-radius:6px;"></div>
              </div>
            </div>"""
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:22px;margin-bottom:20px;">
          {slabel("Class Performance")}
          {rows}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        lane_rows = ""
        for lname, litems, lcolor in LANES:
            lane_rows += f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;
                        background:{BG3};border-radius:10px;padding:11px 14px;">
              <div style="width:9px;height:9px;border-radius:50%;background:{lcolor};flex-shrink:0;"></div>
              <div>
                <div style="font-size:12px;font-weight:600;color:{TEXT};">{lname}</div>
                <div style="font-size:11px;color:{MUTED};margin-top:1px;">{litems}</div>
              </div>
            </div>"""
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:22px;margin-bottom:20px;">
          {slabel("Conveyor Lanes")}
          {lane_rows}
        </div>
        """, unsafe_allow_html=True)

    # Recent history
    if not st.session_state.history:
        hist_body = f"""
        <div style="text-align:center;padding:32px 0;color:{MUTED};">
          <div style="font-size:36px;margin-bottom:10px;opacity:0.35;">🔍</div>
          <div style="font-size:14px;">No classifications yet — go to Waste Identification to start</div>
        </div>"""
    else:
        rows = f"""
        <div style="display:grid;grid-template-columns:40px 1fr 90px 70px 110px;
                    font-size:10px;color:{MUTED};font-weight:700;letter-spacing:1px;text-transform:uppercase;
                    padding-bottom:10px;border-bottom:1px solid {BORDER};margin-bottom:4px;">
          <span>#</span><span>Class</span><span>Confidence</span><span>Bin</span><span>Type</span>
        </div>"""
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            s  = SORTING[h["cls"]]
            rc = ACCENT if s["type"] == "Recyclable" else "#b04040"
            rows += f"""
            <div style="display:grid;grid-template-columns:40px 1fr 90px 70px 110px;
                        font-size:13px;color:{TEXT};padding:9px 0;border-bottom:1px solid {BORDER};">
              <span style="color:{MUTED};font-family:'DM Mono',monospace;">{len(st.session_state.history)-i:02d}</span>
              <span style="text-transform:capitalize;font-weight:500;">{h['cls']}</span>
              <span style="font-family:'DM Mono',monospace;color:{ACCENT};">{h['conf']:.1f}%</span>
              <span style="color:{MUTED};">{s['bin']}</span>
              <span style="color:{rc};font-size:11px;font-weight:600;">{s['type']}</span>
            </div>"""
        hist_body = rows

    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:22px;margin-bottom:20px;">
      {slabel("Recent Classifications")}
      {hist_body}
    </div>
    """, unsafe_allow_html=True)

    if st.button("Go to Waste Identification →", key="goto_wi"):
        st.session_state.page = "Waste Identification"
        st.rerun()

# ════════════════════════════════════════
# WASTE IDENTIFICATION
# ════════════════════════════════════════
elif st.session_state.page == "Waste Identification":

    try:
        pred_sess, gcam_sess = load_models()
        models_ready = True
    except Exception:
        models_ready = False
        st.markdown(f"""
        <div style="background:rgba(176,64,64,0.12);border:1px solid #b04040;border-radius:12px;
                    padding:14px 18px;color:#ff8888;font-size:13px;margin-bottom:20px;">
          ⚠️ Model files not found — place <code>ecosort_model.onnx</code> and
          <code>ecosort_gradcam.onnx</code> in the <code>model/</code> folder.
        </div>
        """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1,1], gap="large")

    with left_col:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:22px;margin-bottom:16px;">
          <div style="font-size:15px;font-weight:700;color:{TEXT};margin-bottom:3px;">Upload Image</div>
          <div style="font-size:12px;color:{MUTED};margin-bottom:14px;">Supports JPG, JPEG, PNG</div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:20px;">
              <div style="font-size:12px;font-weight:600;color:{MUTED};margin-bottom:10px;">Preview</div>
            """, unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown(f"""
              <div style="display:flex;justify-content:space-between;margin-top:10px;">
                <span style="font-size:11px;color:{MUTED};">Dimensions</span>
                <span style="font-size:11px;color:{TEXT};font-family:'DM Mono',monospace;">{img.width} × {img.height} px</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        if not uploaded:
            tags = "".join([
                f'<span style="background:{BG3};color:{MUTED};font-size:11px;'
                f'padding:4px 12px;border-radius:20px;">{c}</span>'
                for c in CLASS_NAMES
            ])
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                        padding:64px 32px;text-align:center;">
              <div style="font-size:44px;margin-bottom:14px;opacity:0.3;">🔍</div>
              <div style="font-size:15px;font-weight:600;color:{TEXT};margin-bottom:8px;">No image uploaded yet</div>
              <div style="font-size:13px;color:{MUTED};max-width:260px;margin:0 auto 20px;line-height:1.6;">
                Upload a waste image on the left to classify it and get a sorting decision
              </div>
              <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)

        elif models_ready and uploaded:
            arr = preprocess(img)
            with st.spinner("Analyzing image..."):
                probs, pred_cls, conf = predict(pred_sess, arr)
                heatmap               = get_gradcam(gcam_sess, arr)
                heat_res, ov          = make_overlay(img, heatmap)
                time.sleep(0.2)

            st.session_state.history.append({"cls": pred_cls, "conf": conf})

            sort      = SORTING[pred_cls]
            icon      = ICONS.get(pred_cls, "♻️")
            rec_color = ACCENT if sort["type"] == "Recyclable" else "#b04040"

            # Classification result
            bars = ""
            for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
                is_top = cls == pred_cls
                bars += f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
                  <span style="width:74px;font-size:12px;color:{''+TEXT if is_top else MUTED};
                               font-weight:{'600' if is_top else '400'};text-transform:capitalize;">{cls}</span>
                  <div style="flex:1;background:{BG3};border-radius:4px;height:5px;overflow:hidden;">
                    <div style="width:{prob*100:.1f}%;height:100%;
                                background:{''+ACCENT if is_top else ACCENT2};border-radius:4px;"></div>
                  </div>
                  <span style="width:38px;font-size:11px;color:{MUTED};text-align:right;
                               font-family:'DM Mono',monospace;">{prob*100:.1f}%</span>
                </div>"""

            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                        padding:22px;margin-bottom:14px;">
              {slabel("Classification Result")}
              <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
                <div style="width:50px;height:50px;border-radius:13px;background:{BG3};
                            display:flex;align-items:center;justify-content:center;
                            font-size:24px;flex-shrink:0;">{icon}</div>
                <div style="flex:1;">
                  <div style="font-size:24px;font-weight:700;color:{TEXT};
                              letter-spacing:-0.5px;text-transform:capitalize;">{pred_cls}</div>
                  <div style="font-size:12px;color:{MUTED};margin-top:1px;">Detected material type</div>
                </div>
                <div style="text-align:right;flex-shrink:0;">
                  <div style="font-size:26px;font-weight:700;color:{ACCENT};
                              font-family:'DM Mono',monospace;">{conf:.1f}%</div>
                  <div style="font-size:11px;color:{MUTED};">confidence</div>
                </div>
              </div>
              <div style="background:{BG3};border-radius:8px;height:6px;overflow:hidden;margin-bottom:18px;">
                <div style="width:{conf:.1f}%;height:100%;background:{ACCENT};border-radius:8px;"></div>
              </div>
              {slabel("All Classes")}
              {bars}
            </div>
            """, unsafe_allow_html=True)

            # Sorting decision + lane pipeline
            lane_boxes = ""
            for lname, litems, lcolor in LANES:
                is_active = sort["lane"] == lname
                if is_active:
                    lane_boxes += f"""
                    <div style="background:{lcolor};border:2px solid {lcolor};border-radius:12px;
                                padding:13px 8px;text-align:center;">
                      <div style="font-size:9px;font-weight:700;color:rgba(255,255,255,0.85);
                                  letter-spacing:0.8px;text-transform:uppercase;margin-bottom:4px;">ACTIVE</div>
                      <div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:3px;">{lname}</div>
                      <div style="font-size:10px;color:rgba(255,255,255,0.75);line-height:1.3;">{litems}</div>
                    </div>"""
                else:
                    lane_boxes += f"""
                    <div style="background:{BG3};border:1px solid {BORDER};border-radius:12px;
                                padding:13px 8px;text-align:center;">
                      <div style="font-size:13px;font-weight:600;color:{MUTED};margin-bottom:3px;">{lname}</div>
                      <div style="font-size:10px;color:{MUTED};opacity:0.65;line-height:1.3;">{litems}</div>
                    </div>"""

            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                        padding:22px;margin-bottom:14px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
                <div style="font-size:10px;font-weight:700;color:{MUTED};letter-spacing:1.4px;text-transform:uppercase;">Sorting Decision</div>
                <div style="background:{'rgba(61,220,132,0.12)' if sort['type']=='Recyclable' else 'rgba(176,64,64,0.12)'};
                            color:{rec_color};font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;">
                  {sort['type']}
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
                <div style="background:{BG3};border-radius:12px;padding:14px;">
                  <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:5px;">Assigned Bin</div>
                  <div style="font-size:16px;font-weight:700;color:{TEXT};">{sort['bin']}</div>
                </div>
                <div style="background:{BG3};border-radius:12px;padding:14px;">
                  <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:5px;">Conveyor Lane</div>
                  <div style="font-size:16px;font-weight:700;color:{TEXT};">{sort['lane']}</div>
                </div>
                <div style="background:{BG3};border-radius:12px;padding:14px;">
                  <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:5px;">Processing Action</div>
                  <div style="font-size:13px;font-weight:600;color:{TEXT};">{sort['action']}</div>
                </div>
                <div style="background:{BG3};border-radius:12px;padding:14px;">
                  <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:5px;">Destination</div>
                  <div style="font-size:12px;font-weight:600;color:{TEXT};">{sort['dest']}</div>
                </div>
              </div>
              {slabel("Conveyor Pipeline")}
              <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
                {lane_boxes}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Grad-CAM
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;padding:22px;">
              {slabel("Grad-CAM Explainability")}
              <div style="font-size:12px;color:{MUTED};margin-top:-8px;margin-bottom:14px;">
                Regions the model focused on to make its decision
              </div>
            """, unsafe_allow_html=True)

            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(f'<div style="font-size:11px;font-weight:600;color:{MUTED};margin-bottom:6px;">Original</div>', unsafe_allow_html=True)
                st.image(np.array(img.resize((224,224))), use_container_width=True)
            with g2:
                st.markdown(f'<div style="font-size:11px;font-weight:600;color:{MUTED};margin-bottom:6px;">Activation Map</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3,3))
                fig.patch.set_facecolor("none")
                ax.imshow(heat_res, cmap="jet"); ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
                buf.seek(0); plt.close()
                st.image(buf, use_container_width=True)
            with g3:
                st.markdown(f'<div style="font-size:11px;font-weight:600;color:{MUTED};margin-bottom:6px;">Overlay</div>', unsafe_allow_html=True)
                st.image(ov, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)