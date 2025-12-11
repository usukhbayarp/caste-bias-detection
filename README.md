# 🇮🇳 Context-Aware AI: Mitigating Caste Bias in Indian Hate Speech Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)  
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace-orange)  
![License](https://img.shields.io/badge/License-MIT-green)

A research project adapting Western AI models to detect **Caste-based Hate Speech** in the Indian digital context.  
This project replicates the **ALBERT baseline** and introduces **Dynamic Identity Masking** to reduce algorithmic bias against marginalized communities (e.g., Dalits, Adivasis).

---

## 📂 Project Structure

```text
├── IndiCASA_dataset - caste.csv        # IndiCASA dataset
├── indicasa_train_seeds.csv            # Filtered high-quality seeds
├── indicasa_final_train_ready.csv      # Augmented & Balanced training data
├── indicasa_test_gold.csv              # Gold standard test set (80 examples)
├── MGSD.csv                            # MGSD dataset, used for baseline
├── SeeGULL - GPT Augmentation.csv      # SeeGULL dataset, used for baseline
├── Winoqueer - GPT Augmentation.csv    # Winoqueer dataset, used for baseline
├── model_output_baseline/              # Replicated ALBERT Baseline (Zero-Shot)
├── model_exp2_albert_indicasa/         # Standard Fine-Tuned Model
├── model_exp3_albert_masking/          # Adapted (Masking) Model (Champion)
├── assignment_notebook.ipynb           # Main experiment notebook (Training & Eval)
├── app.py                              # Streamlit Demo Application
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
```


TODO: convert file structure to follow following structure
```text
├── data/
│   ├── indicasa_train_seeds.csv       # Filtered high-quality seeds
│   ├── indicasa_final_train_ready.csv # Augmented & Balanced training data
│   └── indicasa_test_gold.csv         # Gold standard test set (80 examples)
├── models/
│   ├── model_output_baseline/         # Replicated ALBERT Baseline (Zero-Shot)
│   ├── model_exp2_albert_indicasa/    # Standard Fine-Tuned Model
│   └── model_exp3_albert_masking/     # Adapted (Masking) Model (Champion)
├── notebooks/
│   └── assignment_notebook.ipynb      # Main experiment notebook (Training & Eval)
├── app.py                             # Streamlit Demo Application
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🛠️ Environment Setup

This project uses **Python 3.11**, optimized for **Apple Silicon (M-series)** via MPS, but also supports CUDA and CPU.

### **Clone the Repository**

```bash
git clone https://github.com/usukhbayarp/caste-bias-detection.git
cd caste-bias-detection
```

### **Create Virtual Environment (Conda)**

```bash
conda create -n caste_bias python=3.11
conda activate caste_bias
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Reproduce Results

All experiments are in **assignment_notebook.ipynb**.

---

### **Step 1: Baseline Replication (Zero-Shot)**

**Goal:** Replicate ALBERT baseline on EMGSD dataset.  
**Action:** Run Cells 1–4 in the notebook.  
**Expected Metric:** *Macro F1 ≈ 0.80 (±5%)*.

---

### **Step 2: Dataset Curation (IndiCASA)**
https://github.com/cerai-iitm/IndiCASA

**Goal:** Prepare the Indian-context dataset.  
**Action:** Run “Data Curation” section.

Process includes:

- Loading seeds  
- Filtering for annotator confidence ≥ 3  
- Augmenting with “Hard Negatives” & “Identity Swaps”  
- Class balancing  

**Output files:**

Train: `indicasa_final_train_ready.csv`

Test: `indicasa_test_gold.csv`

---

### **Step 3: Model Adaptation (Training)**

**Models trained:**

- **Experiment 2:** Standard Fine-Tuned (No Masking)  
- **Experiment 3:** Masking Model (40% masking probability)

**Hyperparameters:**

- Batch Size: 16  
- Learning Rate: 2e-5  
- Epochs: 10
- Loss: Weighted Cross-Entropy (Neutral=1.0, Stereotype=2.0)

---

### **Step 4: Evaluation**

Outputs include:

- **McNemar’s Test** — statistical significance between models  
- **SHAP Plots** — feature importance (e.g., impact of “Dalit”)  
- **Bias Audit** — qualitative fairness check  

---

## 🎮 Running the Demo App

The Streamlit dashboard allows real-time testing across all three models.

### Run:

```bash
streamlit run app.py
```

### Features:

- **Live Predictions** (e.g., “The Dalit CEO donated millions”)  
- **Model Comparison** (Baseline vs Standard vs Masking)  
- **LIME Visualizations** for word-level explanations  

---

## 📊 Key Results

| Model                    | Macro F1 | Bias Status     | Key Observation |
|-------------------------|----------|-----------------|-----------------|
| Baseline (Zero-Shot)    | 0.57     | ⚠️ High Bias    | Fails on Indian-specific slurs |
| IndiCASA (Standard)     | 0.71     | ⚠️ Identity Bias | Flags phrases like “Dalit CEO” as toxic |
| IndiCASA + Masking      | 0.78     | ✅ Low Bias      | Correctly handles identity-neutral contexts |

**Statistical Significance:**  
McNemar’s Test (p = 0.21) indicates the accuracy difference between Standard and Masking is not statistically significant, but the reduction in False Positives (Bias) is qualitatively substantial.

---

## 📜 Credits & References

- **Dataset:** IndiCASA (Centre for Responsible AI, IIT Madras)  
- **Base Model:** ALBERT (Hugging Face)  
- **Methodology:** SafeText & Context-Aware Bias Mitigation frameworks  

