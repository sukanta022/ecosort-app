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
    page_title="Eco-Sort · Smart Waste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme'   not in st.session_state: st.session_state.theme   = 'dark'
if 'page'    not in st.session_state: st.session_state.page    = 'Home'
if 'history' not in st.session_state: st.session_state.history = []

T = st.session_state.theme

if T == 'dark':
    bg=      '#0d0f14'; bg2='#13161e'; bg3='#1a1e2a'; bg4='#1f2435'
    border=  'rgba(255,255,255,0.07)'; text='#e4e8f4'; muted='#6b7390'
    accent=  '#3ddc84'; accent2='#38bdf8'; danger='#f87171'
    shadow=  '0 2px 20px rgba(0,0,0,0.55)'
else:
    bg=      '#f0f2f7'; bg2='#ffffff'; bg3='#e8ecf4'; bg4='#dde2ed'
    border=  'rgba(0,0,0,0.08)'; text='#111827'; muted='#6b7280'
    accent=  '#16a34a'; accent2='#0284c7'; danger='#dc2626'
    shadow=  '0 2px 16px rgba(0,0,0,0.08)'

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
.stApp{{background:{bg}!important;font-family:'Sora',sans-serif!important;color:{text}!important;}}
#MainMenu,footer,header{{visibility:hidden!important;}}
.block-container{{padding:0!important;max-width:100%!important;}}
section[data-testid="stSidebar"]{{background:{bg2}!important;border-right:1px solid {border}!important;min-width:252px!important;max-width:252px!important;}}
section[data-testid="stSidebar"]>div{{padding:0!important;background:{bg2}!important;}}
section[data-testid="stSidebar"] *{{font-family:'Sora',sans-serif!important;color:{text}!important;}}
section[data-testid="stSidebar"] .stButton>button{{background:transparent!important;color:{muted}!important;border:none!important;border-radius:10px!important;font-size:13.5px!important;font-weight:400!important;padding:10px 14px!important;text-align:left!important;width:100%!important;transition:all 0.15s!important;margin-bottom:2px!important;}}
section[data-testid="stSidebar"] .stButton>button:hover{{background:{bg3}!important;color:{text}!important;}}
.main-area .stButton>button{{background:{accent}!important;color:{'#0d1a0d' if T=='dark' else '#fff'}!important;border:none!important;border-radius:10px!important;font-family:'Sora',sans-serif!important;font-weight:600!important;font-size:13px!important;padding:10px 22px!important;transition:opacity 0.2s!important;}}
.main-area .stButton>button:hover{{opacity:0.85!important;}}
[data-testid="stFileUploader"]{{background:{bg3}!important;border:2px dashed {border}!important;border-radius:14px!important;}}
[data-testid="stFileUploader"]:hover{{border-color:{accent}!important;}}
[data-testid="stFileUploadDropzone"]{{background:transparent!important;}}
[data-testid="stFileUploader"] label{{color:{muted}!important;font-size:13px!important;}}
[data-testid="stImage"] img{{border-radius:12px!important;}}
.stSpinner>div{{border-top-color:{accent}!important;}}
[data-testid="stToolbar"]{{display:none!important;}}
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{border};border-radius:4px;}}
</style>
""", unsafe_allow_html=True)

CLASS_NAMES  = ['cardboard','glass','metal','paper','plastic','trash']
PREDICT_PATH = 'model/ecosort_model.onnx'
GRADCAM_PATH = 'model/ecosort_gradcam.onnx'

SORTING = {
    'cardboard': {'bin':'Bin A','lane':'Lane 1','lane_num':1,'action':'Flatten & Compress',    'dest':'Paper Recycling Facility', 'recyclable':True, 'color':'#b08050'},
    'glass':     {'bin':'Bin B','lane':'Lane 2','lane_num':2,'action':'Colour Sort & Crush',   'dest':'Glass Processing Plant',   'recyclable':True, 'color':'#4a90c4'},
    'metal':     {'bin':'Bin C','lane':'Lane 3','lane_num':3,'action':'Magnetic Separation',   'dest':'Metal Scrap Facility',     'recyclable':True, 'color':'#7a8fa0'},
    'paper':     {'bin':'Bin D','lane':'Lane 1','lane_num':1,'action':'Stack & Bundle',        'dest':'Paper Recycling Facility', 'recyclable':True, 'color':'#6aaa6a'},
    'plastic':   {'bin':'Bin E','lane':'Lane 4','lane_num':4,'action':'Shred & Pelletize',     'dest':'Plastic Recycling Plant',  'recyclable':True, 'color':'#d4903a'},
    'trash':     {'bin':'Bin F','lane':'Lane 5','lane_num':5,'action':'Compact & Seal',        'dest':'Controlled Landfill',      'recyclable':False,'color':'#c05050'},
}
LANES = {
    1:{'label':'Lane 1','materials':'Cardboard / Paper','classes':['cardboard','paper'],'color':'#6aaa6a'},
    2:{'label':'Lane 2','materials':'Glass',            'classes':['glass'],            'color':'#4a90c4'},
    3:{'label':'Lane 3','materials':'Metal',            'classes':['metal'],            'color':'#7a8fa0'},
    4:{'label':'Lane 4','materials':'Plastic',          'classes':['plastic'],          'color':'#d4903a'},
    5:{'label':'Lane 5','materials':'Trash',            'classes':['trash'],            'color':'#c05050'},
}
CLASS_ICON = {'cardboard':'📦','glass':'🫙','metal':'🔩','paper':'📄','plastic':'🧴','trash':'🗑️'}

@st.cache_resource
def load_models():
    return ort.InferenceSession(PREDICT_PATH), ort.InferenceSession(GRADCAM_PATH)

def preprocess(img):
    arr = np.array(img.resize((224,224))).astype(np.float32)/255.0
    return np.expand_dims(arr,0)

def run_predict(sess,arr):
    n=sess.get_inputs()[0].name; preds=sess.run(None,{n:arr})[0][0]
    return preds,CLASS_NAMES[np.argmax(preds)],float(np.max(preds))*100

def run_gradcam(sess,arr):
    n=sess.get_inputs()[0].name; out=sess.run(None,{n:arr})
    conv=out[0][0]; pred=out[1][0]; top=np.argmax(pred)
    g=np.zeros_like(conv)
    for i in range(conv.shape[-1]): g[:,:,i]=conv[:,:,i].mean()*pred[top]
    w=np.mean(g,axis=(0,1)); h=np.zeros(conv.shape[:2],dtype=np.float32)
    for i,ww in enumerate(w): h+=ww*conv[:,:,i]
    h=np.maximum(h,0)
    if h.max()>0: h/=h.max()
    return h

def make_overlay(img,heatmap):
    orig=np.array(img.resize((224,224))); hr=cv2.resize(heatmap,(224,224))
    hc=cv2.applyColorMap(np.uint8(255*hr),cv2.COLORMAP_JET)
    hc=cv2.cvtColor(hc,cv2.COLOR_BGR2RGB)
    return hr, cv2.addWeighted(orig,0.6,hc,0.4,0)

def heatmap_png(hr):
    fig,ax=plt.subplots(figsize=(3,3)); fig.patch.set_alpha(0)
    ax.imshow(hr,cmap='jet'); ax.axis('off')
    buf=io.BytesIO(); plt.savefig(buf,format='png',bbox_inches='tight',pad_inches=0,transparent=True)
    buf.seek(0); plt.close(); return buf

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div style="padding:26px 20px 20px;">
      <div style="display:flex;align-items:center;gap:11px;">
        <div style="width:36px;height:36px;background:{accent};border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:{'#0d1a0d' if T=='dark' else '#fff'};flex-shrink:0;">♻</div>
        <div>
          <div style="font-size:16px;font-weight:700;color:{text};letter-spacing:-0.3px;line-height:1.2;">Eco-Sort</div>
          <div style="font-size:10.5px;color:{muted};font-family:'JetBrains Mono',monospace;">v1.0 · Smart Waste AI</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:{border};margin:0 16px 18px;"></div>
    <div style="padding:0 12px;">
      <div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.8px;text-transform:uppercase;padding:0 6px;margin-bottom:8px;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    for pname in ['Home','Waste Identification']:
        active = st.session_state.page == pname
        if active:
            st.markdown(f"""
            <div style="margin:0 12px 2px;background:{bg3};border-radius:10px;padding:10px 14px;border-left:3px solid {accent};">
              <span style="font-size:13.5px;font-weight:600;color:{text};">{pname}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.sidebar.button(pname, key=f"nav_{pname}", use_container_width=True):
                st.session_state.page = pname; st.rerun()

    st.markdown(f"""
    <div style="height:1px;background:{border};margin:18px 16px;"></div>
    <div style="padding:0 20px;">
      <div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.8px;text-transform:uppercase;margin-bottom:12px;">System Status</div>
      {''.join([f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;"><div style="width:7px;height:7px;border-radius:50%;background:{accent};box-shadow:0 0 7px {accent}77;flex-shrink:0;"></div><span style="font-size:12.5px;color:{text};">{s}</span></div>' for s in ['Model loaded','MobileNetV2 ready','Grad-CAM active']])}
    </div>
    <div style="height:1px;background:{border};margin:18px 16px;"></div>
    <div style="padding:0 20px;">
      <div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.8px;text-transform:uppercase;margin-bottom:12px;">Session Stats</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="font-size:12.5px;color:{muted};">Classifications</span>
        <span style="font-size:13px;font-weight:600;color:{text};font-family:'JetBrains Mono',monospace;">{len(st.session_state.history):02d}</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="font-size:12.5px;color:{muted};">Model Accuracy</span>
        <span style="font-size:13px;font-weight:600;color:{accent};font-family:'JetBrains Mono',monospace;">86.77%</span>
      </div>
    </div>
    <div style="height:1px;background:{border};margin:18px 16px 16px;"></div>
    <div style="padding:0 14px 8px;">
      <div style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1.8px;text-transform:uppercase;margin-bottom:10px;">Appearance</div>
    </div>
    """, unsafe_allow_html=True)

    tc1,tc2 = st.columns(2)
    with tc1:
        if st.button("☀ Light", key="btn_light", use_container_width=True):
            st.session_state.theme='light'; st.rerun()
    with tc2:
        if st.button("◑ Dark", key="btn_dark", use_container_width=True):
            st.session_state.theme='dark'; st.rerun()

# ── Main ──
with st.container():
    st.markdown('<div class="main-area">', unsafe_allow_html=True)

    page_sub = {
        'Home':'System overview, model performance, and session history',
        'Waste Identification':'Upload a waste image to classify and receive sorting instructions',
    }
    st.markdown(f"""
    <div style="background:{bg2};border-bottom:1px solid {border};padding:18px 32px;display:flex;align-items:center;justify-content:space-between;">
      <div>
        <div style="font-size:19px;font-weight:700;color:{text};letter-spacing:-0.4px;">{st.session_state.page}</div>
        <div style="font-size:12px;color:{muted};margin-top:2px;">{page_sub[st.session_state.page]}</div>
      </div>
      <div style="background:{'rgba(61,220,132,0.12)' if T=='dark' else 'rgba(22,163,74,0.10)'};color:{accent};font-size:11px;font-weight:600;padding:5px 14px;border-radius:20px;font-family:'JetBrains Mono',monospace;">MobileNetV2 · 86.77%</div>
    </div>
    <div style="padding:28px 32px 40px;">
    """, unsafe_allow_html=True)

    # ════════════ HOME ════════════
    if st.session_state.page == 'Home':

        # Overview banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{'#13261a' if T=='dark' else '#e8f5ee'} 0%,{bg2} 65%);border:1px solid {'rgba(61,220,132,0.2)' if T=='dark' else 'rgba(22,163,74,0.15)'};border-radius:18px;padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden;">
          <div style="position:absolute;right:28px;top:50%;transform:translateY(-50%);font-size:80px;opacity:0.07;line-height:1;pointer-events:none;">♻</div>
          <div style="font-size:10.5px;font-weight:600;color:{accent};letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">About Eco-Sort</div>
          <div style="font-size:21px;font-weight:700;color:{text};letter-spacing:-0.5px;margin-bottom:10px;line-height:1.35;">Intelligent Waste Classification<br>&amp; Automated Sorting System</div>
          <div style="font-size:13px;color:{muted};line-height:1.75;max-width:620px;">
            Eco-Sort uses deep learning (MobileNetV2 with Transfer Learning) to classify waste images into
            six categories — Cardboard, Glass, Metal, Paper, Plastic, and Trash. Each classification is
            paired with an automated sorting decision that assigns the item to the correct conveyor lane
            and processing facility. Grad-CAM visualisation provides transparent, human-readable
            explanations for every prediction.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Stat cards
        sc1,sc2,sc3,sc4 = st.columns(4)
        for col,(val,label,color,sub) in zip([sc1,sc2,sc3,sc4],[
            ('86.77%','Test Accuracy',accent,'MobileNetV2 · Phase 1'),
            ('6','Waste Classes',accent2,'Cardboard → Trash'),
            ('0.85','F1-Score','#a78bfa','Weighted average'),
            ('5','Sorting Lanes','#fb923c','Dedicated routes'),
        ]):
            with col:
                st.markdown(f"""
                <div style="background:{bg2};border:1px solid {border};border-radius:14px;padding:20px;position:relative;overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{color};border-radius:14px 14px 0 0;"></div>
                  <div style="font-size:30px;font-weight:700;color:{text};font-family:'JetBrains Mono',monospace;letter-spacing:-1px;line-height:1;margin-bottom:6px;">{val}</div>
                  <div style="font-size:13px;font-weight:600;color:{text};margin-bottom:3px;">{label}</div>
                  <div style="font-size:11px;color:{muted};">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        cl,cr = st.columns([3,2], gap="large")
        with cl:
            st.markdown(f"""
            <div style="background:{bg2};border:1px solid {border};border-radius:14px;padding:22px 24px;">
              <div style="font-size:14px;font-weight:700;color:{text};margin-bottom:18px;">Classification Performance by Class</div>
            """, unsafe_allow_html=True)
            for cls,score,color in [('cardboard',0.93,'#b08050'),('paper',0.91,'#6aaa6a'),
                                     ('glass',0.83,'#4a90c4'),('metal',0.82,'#7a8fa0'),
                                     ('plastic',0.81,'#d4903a'),('trash',0.69,'#c05050')]:
                st.markdown(f"""
                <div style="margin-bottom:13px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                    <span style="font-size:13px;color:{text};font-weight:500;text-transform:capitalize;">{cls}</span>
                    <span style="font-size:12px;color:{muted};font-family:'JetBrains Mono',monospace;">F1 = {score}</span>
                  </div>
                  <div style="background:{bg3};border-radius:6px;height:8px;overflow:hidden;">
                    <div style="width:{int(score*100)}%;height:100%;background:{color};border-radius:6px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cr:
            st.markdown(f"""
            <div style="background:{bg2};border:1px solid {border};border-radius:14px;padding:22px 24px;">
              <div style="font-size:14px;font-weight:700;color:{text};margin-bottom:18px;">Sorting Lane Assignment</div>
            """, unsafe_allow_html=True)
            for ln,info in LANES.items():
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:11px 14px;background:{bg3};border-radius:10px;margin-bottom:8px;">
                  <div style="width:28px;height:28px;border-radius:8px;background:{info['color']}22;border:1px solid {info['color']}55;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                    <div style="width:8px;height:8px;border-radius:50%;background:{info['color']};"></div>
                  </div>
                  <div>
                    <div style="font-size:12.5px;font-weight:600;color:{text};">{info['label']}</div>
                    <div style="font-size:11px;color:{muted};">{info['materials']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        # History table
        st.markdown(f"""
        <div style="background:{bg2};border:1px solid {border};border-radius:14px;padding:22px 24px;">
          <div style="font-size:14px;font-weight:700;color:{text};margin-bottom:18px;">Recent Classifications</div>
        """, unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown(f"""
            <div style="text-align:center;padding:36px;">
              <div style="font-size:36px;opacity:0.2;margin-bottom:10px;">🔍</div>
              <div style="font-size:13px;color:{muted};">No classifications yet — go to <strong>Waste Identification</strong> to start.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:40px 1fr 120px 100px 140px;padding:0 4px 10px;border-bottom:1px solid {border};margin-bottom:6px;">
              {''.join([f'<span style="font-size:10px;font-weight:600;color:{muted};letter-spacing:1px;text-transform:uppercase;">{h}</span>' for h in ['#','Class','Confidence','Bin','Status']])}
            </div>
            """, unsafe_allow_html=True)
            for i,h in enumerate(reversed(st.session_state.history[-10:])):
                s=SORTING[h['cls']]; sc=accent if s['recyclable'] else danger
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:40px 1fr 120px 100px 140px;padding:9px 4px;border-bottom:1px solid {border};align-items:center;">
                  <span style="font-size:12px;color:{muted};font-family:'JetBrains Mono',monospace;">{len(st.session_state.history)-i:02d}</span>
                  <span style="font-size:13px;font-weight:500;color:{text};text-transform:capitalize;">{h['cls']}</span>
                  <span style="font-size:12px;color:{accent};font-family:'JetBrains Mono',monospace;">{h['conf']:.1f}%</span>
                  <span style="font-size:12px;color:{muted};">{s['bin']}</span>
                  <span style="font-size:11px;font-weight:600;color:{sc};background:{'rgba(61,220,132,0.1)' if s['recyclable'] else 'rgba(248,113,113,0.1)'};padding:3px 10px;border-radius:20px;display:inline-block;">{'Recyclable' if s['recyclable'] else 'Non-Recyclable'}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        if st.button("Open Waste Identification →"):
            st.session_state.page='Waste Identification'; st.rerun()

    # ════════════ WASTE IDENTIFICATION ════════════
    elif st.session_state.page == 'Waste Identification':

        try:
            pred_sess,gcam_sess = load_models(); models_ok=True
        except:
            models_ok=False
            st.markdown(f"""
            <div style="background:rgba(220,38,38,0.08);border:1px solid {danger};border-radius:12px;padding:14px 18px;margin-bottom:20px;">
              <span style="font-size:13px;color:{danger};font-weight:500;">Model files not found — place <code>ecosort_model.onnx</code> and <code>ecosort_gradcam.onnx</code> in the <code>model/</code> folder.</span>
            </div>
            """, unsafe_allow_html=True)

        # Upload
        up_col, _ = st.columns([2,1])
        with up_col:
            st.markdown(f'<div style="font-size:14px;font-weight:700;color:{text};margin-bottom:12px;">Upload Waste Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Drag & drop or click to browse — JPG, JPEG, PNG", type=['jpg','jpeg','png'], label_visibility="visible")

        if not uploaded:
            st.markdown(f"""
            <div style="background:{bg2};border:1px solid {border};border-radius:16px;padding:64px 32px;text-align:center;margin-top:16px;">
              <div style="font-size:52px;opacity:0.18;margin-bottom:16px;">♻</div>
              <div style="font-size:16px;font-weight:600;color:{text};margin-bottom:8px;">No image uploaded yet</div>
              <div style="font-size:13px;color:{muted};line-height:1.7;max-width:380px;margin:0 auto;">Upload a waste image above to receive an AI-powered classification, sorting decision, and Grad-CAM visual explanation.</div>
              <div style="margin-top:22px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
                {''.join([f'<span style="background:{bg3};color:{muted};font-size:11.5px;font-weight:500;padding:5px 14px;border-radius:20px;text-transform:capitalize;">{c}</span>' for c in CLASS_NAMES])}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            img=Image.open(uploaded).convert('RGB'); arr=preprocess(img)
            with st.spinner("Running classification pipeline…"):
                probs,pred_cls,conf = run_predict(pred_sess,arr)
                heatmap             = run_gradcam(gcam_sess,arr)
                hr,ov               = make_overlay(img,heatmap)
                heat_buf            = heatmap_png(hr)
            st.session_state.history.append({'cls':pred_cls,'conf':conf})
            sort=SORTING[pred_cls]; rc=accent if sort['recyclable'] else danger
            rl='Recyclable' if sort['recyclable'] else 'Non-Recyclable'

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Row 1: Image + Result ──
            ic,rc_col = st.columns([1,1], gap="large")
            with ic:
                st.markdown(f"""
                <div style="background:{bg2};border:1px solid {border};border-radius:16px;padding:20px 22px;">
                  <div style="font-size:10.5px;font-weight:600;color:{muted};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">Input Image</div>
                """, unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(f"""
                  <div style="display:flex;justify-content:space-between;margin-top:12px;padding-top:12px;border-top:1px solid {border};">
                    <span style="font-size:11.5px;color:{muted};">Filename</span>
                    <span style="font-size:11.5px;color:{text};font-family:'JetBrains Mono',monospace;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{uploaded.name}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:8px;">
                    <span style="font-size:11.5px;color:{muted};">Dimensions</span>
                    <span style="font-size:11.5px;color:{text};font-family:'JetBrains Mono',monospace;">{img.width} × {img.height} px</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with rc_col:
                st.markdown(f"""
                <div style="background:{bg2};border:1px solid {border};border-radius:16px;padding:20px 22px;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
                    <div style="font-size:10.5px;font-weight:600;color:{muted};letter-spacing:1.5px;text-transform:uppercase;">Classification Result</div>
                    <div style="background:{'rgba(61,220,132,0.1)' if sort['recyclable'] else 'rgba(248,113,113,0.1)'};color:{rc};font-size:11px;font-weight:600;padding:4px 12px;border-radius:20px;">{rl}</div>
                  </div>
                  <div style="display:flex;align-items:center;gap:14px;margin-bottom:22px;">
                    <div style="width:56px;height:56px;border-radius:14px;background:{bg3};display:flex;align-items:center;justify-content:center;font-size:28px;flex-shrink:0;">{CLASS_ICON[pred_cls]}</div>
                    <div style="flex:1;">
                      <div style="font-size:28px;font-weight:700;color:{text};letter-spacing:-0.5px;text-transform:capitalize;line-height:1.1;">{pred_cls}</div>
                      <div style="font-size:12px;color:{muted};margin-top:3px;">Detected material type</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:30px;font-weight:700;color:{accent};font-family:'JetBrains Mono',monospace;line-height:1;">{conf:.1f}%</div>
                      <div style="font-size:11px;color:{muted};margin-top:2px;">confidence</div>
                    </div>
                  </div>
                  <div style="background:{bg3};border-radius:8px;height:6px;overflow:hidden;margin-bottom:22px;">
                    <div style="width:{conf:.1f}%;height:100%;background:{accent};border-radius:8px;"></div>
                  </div>
                  <div style="font-size:10.5px;font-weight:600;color:{muted};letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;">Probability Distribution</div>
                """, unsafe_allow_html=True)
                for cls,prob in sorted(zip(CLASS_NAMES,probs),key=lambda x:x[1],reverse=True):
                    it=cls==pred_cls; bc=accent if it else (accent2 if prob>0.05 else bg4)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px;">
                      <span style="width:72px;font-size:12px;color:{text if it else muted};font-weight:{'600' if it else '400'};text-transform:capitalize;">{cls}</span>
                      <div style="flex:1;background:{bg3};border-radius:5px;height:5px;overflow:hidden;">
                        <div style="width:{prob*100:.1f}%;height:100%;background:{bc};border-radius:5px;"></div>
                      </div>
                      <span style="width:42px;font-size:11px;color:{muted};text-align:right;font-family:'JetBrains Mono',monospace;">{prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # ── Row 2: Sorting decision ──
            st.markdown(f"""
            <div style="background:{bg2};border:1px solid {border};border-radius:16px;padding:22px 24px;margin-bottom:20px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
                <div>
                  <div style="font-size:10.5px;font-weight:600;color:{muted};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">Automated Sorting Decision</div>
                  <div style="font-size:15px;font-weight:600;color:{text};">
                    Routed to <span style="color:{accent};">{sort['bin']}</span> via <span style="color:{accent};">{sort['lane']}</span>
                  </div>
                </div>
                <div style="background:{'rgba(61,220,132,0.1)' if sort['recyclable'] else 'rgba(248,113,113,0.1)'};color:{rc};font-size:11.5px;font-weight:600;padding:6px 16px;border-radius:20px;">{rl}</div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
              {''.join([f"""
                <div style="background:{bg3};border-radius:12px;padding:14px 16px;">
                  <div style="font-size:9.5px;font-weight:600;color:{muted};letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">{label}</div>
                  <div style="font-size:14px;font-weight:600;color:{text};line-height:1.3;">{val}</div>
                </div>""" for label,val in [('Assigned Bin',sort['bin']),('Conveyor Lane',sort['lane']),('Processing Action',sort['action']),('Destination',sort['dest'])]])}
              </div>
              <div style="font-size:10.5px;font-weight:600;color:{muted};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:14px;">Conveyor Lane Pipeline — Active Route Highlighted</div>
              <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;">
            """, unsafe_allow_html=True)

            for ln_num,ln_info in LANES.items():
                active = sort['lane_num']==ln_num
                if active:
                    st.markdown(f"""
                    <div style="background:{ln_info['color']}18;border:2px solid {ln_info['color']};border-radius:14px;padding:18px 12px;text-align:center;">
                      <div style="width:12px;height:12px;border-radius:50%;background:{ln_info['color']};margin:0 auto 10px;box-shadow:0 0 10px {ln_info['color']}99;"></div>
                      <div style="font-size:13px;font-weight:700;color:{text};margin-bottom:5px;">{ln_info['label']}</div>
                      <div style="font-size:11px;color:{muted};margin-bottom:10px;line-height:1.4;">{ln_info['materials']}</div>
                      <div style="background:{ln_info['color']};color:#fff;font-size:10px;font-weight:700;padding:3px 10px;border-radius:10px;display:inline-block;letter-spacing:0.5px;">ACTIVE</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:{bg3};border:1.5px solid {border};border-radius:14px;padding:18px 12px;text-align:center;opacity:0.65;">
                      <div style="width:10px;height:10px;border-radius:50%;background:{muted};margin:0 auto 10px;"></div>
                      <div style="font-size:13px;font-weight:500;color:{muted};margin-bottom:5px;">{ln_info['label']}</div>
                      <div style="font-size:11px;color:{muted};line-height:1.4;">{ln_info['materials']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

            # ── Row 3: Grad-CAM ──
            st.markdown(f"""
            <div style="background:{bg2};border:1px solid {border};border-radius:16px;padding:22px 24px;">
              <div style="margin-bottom:18px;">
                <div style="font-size:10.5px;font-weight:600;color:{muted};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">Grad-CAM Explainability</div>
                <div style="font-size:13px;color:{muted};line-height:1.7;max-width:680px;">
                  The activation heatmap shows which regions of the image the model focused on.
                  Warmer colours (red/orange) indicate higher model attention when making its decision.
                </div>
              </div>
            """, unsafe_allow_html=True)

            gc1,gc2,gc3 = st.columns(3, gap="medium")
            for col,lbl,src in zip([gc1,gc2,gc3],
                                   ['Original Input','Activation Heatmap','Overlay'],
                                   [np.array(img.resize((224,224))),heat_buf,ov]):
                with col:
                    st.markdown(f"""
                    <div style="background:{bg3};border-radius:12px;padding:14px 14px 10px;">
                      <div style="font-size:10.5px;font-weight:600;color:{muted};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">{lbl}</div>
                    """, unsafe_allow_html=True)
                    st.image(src, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)