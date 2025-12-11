import streamlit as st
from transformers import pipeline
import pandas as pd
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
import os

# --- Page Config ---
st.set_page_config(page_title="Caste Bias Auditor", layout="wide")

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'audit_results' not in st.session_state:
    st.session_state.audit_results = None

# --- Sidebar: History ---
with st.sidebar:
    st.header("📜 Audit History")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, hide_index=True) 
    else:
        st.caption("No history yet.")

# --- Main Header ---
st.title("🇮🇳 AI Caste Bias Auditor")
st.markdown("""
**Context:** Auditing AI models for Caste Bias in the Indian context.
""")

# --- Load Models ---
@st.cache_resource
def load_pipelines():
    # Detect Hardware
    if torch.backends.mps.is_available():
        device = "mps"
        device_name = "Mac GPU"
    elif torch.cuda.is_available():
        device = 0
        device_name = "NVIDIA GPU"
    else:
        device = -1
        device_name = "CPU"
        
    # SPECIFIC PATHS to Models
    path_baseline = "model_output_baseline/emgsd_replicated"
    path_standard = "model_exp2_albert_indicasa/final_model"
    path_masking = "model_exp3_albert_masking/final_model"
    
    try:
        # Load pipelines (top_k=None ensures we get scores for both labels)
        pipe_1 = pipeline("text-classification", model=path_baseline, tokenizer=path_baseline, top_k=None, device=device)
        pipe_2 = pipeline("text-classification", model=path_standard, tokenizer=path_standard, top_k=None, device=device)
        pipe_3 = pipeline("text-classification", model=path_masking, tokenizer=path_masking, top_k=None, device=device)
        
        return pipe_1, pipe_2, pipe_3, device_name
    except Exception as e:
        return None, None, None, str(e)

with st.spinner("Loading AI Models... (This may take a minute)"):
    pipe_baseline, pipe_standard, pipe_masking, device_used = load_pipelines()

if not pipe_baseline:
    st.error(f"Failed to load models! Error: {device_used}")
    st.stop()
else:
    st.caption(f"Running on: {device_used}")

# --- LIME Helper ---
def run_lime(text, pipe):
    def predictor(texts):
        # LIME needs raw numpy array of probabilities
        outputs = pipe(texts, truncation=True, max_length=128)
        # Ensure consistent order: [Neutral Score, Stereotype Score]
        # Assuming label '0' is Neutral and '1' is Stereotype
        scores = []
        for out in outputs:
            # Sort by label ID to ensure column 0 is always Label 0
            sorted_out = sorted(out, key=lambda x: x['label'])
            scores.append([item['score'] for item in sorted_out])
        return np.array(scores)
    
    # Class names must match the order in 'scores' above
    explainer = LimeTextExplainer(class_names=["Neutral", "Stereotype"])
    exp = explainer.explain_instance(text, predictor, num_features=6)
    return exp.as_html()

# --- Helper Labels ---
def get_status(prediction):
    # 'prediction' is a list of dicts: [{'label': 'LABEL_0', 'score': 0.9}, ...]
    # Find the one with highest score
    top_pred = sorted(prediction, key=lambda x: x['score'], reverse=True)[0]
    label = top_pred['label']
    score = top_pred['score']
    
    # Map label to status
    # Assuming LABEL_1 = Stereotype, LABEL_0 = Neutral
    if "LABEL_1" in label or "Stereotype" in label or label == 1:
        return "Stereotype ⚠️", "error", score
    else:
        return "Neutral ✅", "success", score

# --- Input Section ---
st.subheader("Test a Sentence")
default_text = "The Dalit CEO donated millions to charity."
user_input = st.text_input("Enter text here:", value=default_text)

# --- EXECUTION LOGIC (PERSISTENT) ---
if st.button("Run Audit", type="primary"):
    if not user_input.strip():
        st.warning("Please enter text.")
        st.stop()

    # 1. Run Predictions
    raw_1 = pipe_baseline(user_input)[0]
    raw_2 = pipe_standard(user_input)[0]
    raw_3 = pipe_masking(user_input)[0]
    
    status_1, color_1, score_1 = get_status(raw_1)
    status_2, color_2, score_2 = get_status(raw_2)
    status_3, color_3, score_3 = get_status(raw_3)

    # 2. Store in Session State
    st.session_state.audit_results = {
        "input": user_input,
        "base": (status_1, color_1, score_1),
        "std": (status_2, color_2, score_2),
        "mask": (status_3, color_3, score_3)
    }

    # 3. Update History
    new_entry = {
        "Text": user_input,
        "Base": status_1,
        "Std": status_2,
        "Mask": status_3
    }
    st.session_state.history.insert(0, new_entry)
    # Keep history short
    st.session_state.history = st.session_state.history[:10]

# --- RENDER RESULTS (From State) ---
if st.session_state.audit_results:
    res = st.session_state.audit_results
    
    # Sync Warning
    if res["input"] != user_input:
        st.warning("⚠️ Input changed. Click 'Run Audit' to update results.")

    # Columns for 3 Models
    col1, col2, col3 = st.columns(3)

    with col1:
        s, c, sc = res["base"]
        # Case 1: Baseline Albert
        st.info("1. Baseline Albert")
        st.metric("Confidence", f"{sc:.1%}")
        if c == "error": st.error(s)
        else: st.success(s)

    with col2:
        s, c, sc = res["std"]
        # Case 2: Baseline Albert + indiCASA
        st.warning("2. Baseline Albert + indiCASA")
        st.metric("Confidence", f"{sc:.1%}")
        if c == "error": st.error(s)
        else: st.success(s)

    with col3:
        s, c, sc = res["mask"]
        # Case 3: Baseline Albert + indiCASA + Masking
        st.success("3. Baseline Albert + indiCASA + Masking")
        st.metric("Confidence", f"{sc:.1%}")
        if c == "error": st.error(s)
        else: st.success(s)

    # 4. Explainability
    st.divider()
    with st.expander("🔍 Deep Explanation (LIME)"):
        st.info("Comparing: (2) Baseline Albert + indiCASA vs. (3) Baseline Albert + indiCASA + Masking")
        
        if st.button("Generate Visualization"):
            with st.spinner("Calculating feature importance... (This takes 10-20s)"):
                try:
                    lime_col1, lime_col2 = st.columns(2)
                    
                    # 1. Explain Standard (IndiCASA)
                    with lime_col1:
                        st.markdown("**2. Baseline Albert + indiCASA:**")
                        html_2 = run_lime(res["input"], pipe_standard)
                        components.html(html_2, height=200, scrolling=True)
                        
                    # 2. Explain Masking (The Best Model)
                    with lime_col2:
                        st.markdown("**3. Baseline Albert + indiCASA + Masking:**")
                        html_3 = run_lime(res["input"], pipe_masking)
                        components.html(html_3, height=200, scrolling=True)
                        
                except Exception as e:
                    st.error(f"LIME Error: {e}")