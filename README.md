# Child Maintenance Estimator 👶

This repository contains the **Child Maintenance Estimator**, a Streamlit-based tool and underlying data science model designed to estimate **monthly child maintenance awards** based on parental income and children’s details.  

The estimator uses a **calibrated per-child banded model** that outputs **ranges** (rounded to the nearest $10) to reflect realistic outcomes and avoid false precision.  

---

## 📖 What this tool does
- Estimates monthly child maintenance award ranges.  
- Inputs required:  
  - Father’s monthly income  
  - Mother’s monthly income  
  - Ages of up to 4 children  
  - Exception flags (if applicable: child ≥21 but in school, NS, or disabled)  
- Outputs: Predicted **range of maintenance award**.  
- ⚠️ Currently limited to **4 children**. Very few real cases involve >4 children; predictions for such cases would be less accurate. Future updates may extend support.

---

## 🗂 Repository structure
- `app.py` → Streamlit web application  
- `models/model_per_child_v2_calibrated_banded.joblib` → Final trained model  
- `MODEL_CARD.md` → Full documentation of methodology, data, ethical considerations  
- `requirements.txt` → Python dependencies  

---

## ⚙️ Installation and setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/child-maintenance-calculator.git
cd child-maintenance-calculator
```

### 2. Install dependencies
It is recommended to use a virtual environment.  
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will launch in your browser (default: http://localhost:8501).

---

## 🧮 Model details
- **Model type:** Per-child calibrated regression with banding + rounding  
- **Version:** v2 (August 2025)  
- **Training data:** 69 clean real cases (2020–2023) + ML-synthetic augmentation  
- **Wrapper class:** `CMPerChildModelRounded`  
- **Output:** Ranges rounded to nearest $10  

See [`MODEL_CARD.md`](MODEL_CARD.md) for full details on training data, cleaning, synthetic augmentation, and evaluation.

---

## 🧑‍⚖️ Intended use
- Designed as a **decision-support tool** for exploration and policy analysis.  
- Not legal advice. Actual court orders may differ due to case-specific circumstances.  

---

## 🔎 Validation
A **blind expert review** process is set up: experts review predicted ranges without seeing the true awards, ensuring unbiased feedback. Metrics: capture rate, band sharpness, anomaly detection.

---

## 📜 License
To be determined by the owning organisation.  

---

## 📚 Further reading
- [MODEL_CARD.md](MODEL_CARD.md) for detailed methodology and governance.  
