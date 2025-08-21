# Model Card: Child Maintenance Estimator

## Table of Contents
- [Model Details](#model-details)
- [Intended Use](#intended-use)
- [Training Data](#training-data)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Synthetic Data Strategy](#synthetic-data-strategy)
- [Model Development & Evolution](#model-development--evolution)
- [Calibration, Banding & Rounding](#calibration-banding--rounding)
- [Evaluation & Validation](#evaluation--validation)
- [Ethical Considerations & Limitations](#ethical-considerations--limitations)
- [Deployment](#deployment)
- [Maintenance & Versioning](#maintenance--versioning)

---

## Model Details
- **Name:** Child Maintenance Estimator  
- **Version:** `model_per_child_v2_calibrated_banded.joblib`  
- **Wrapper Class:** `CMPerChildModelRounded`  
- **Repository:** Included in this repo under `/models/`  

The model estimates a **range of monthly child maintenance awards** based on:
- Father’s income  
- Mother’s income  
- Number and ages of children (up to 4 children in current UI)  
- Exception criteria (children ≥21 in school, NS, or disabled)  

---

## Intended Use
This estimator is designed as a **decision-support tool** for exploring possible maintenance awards under Singapore’s family law context.  

- **Intended Users:** Legal officers, policy analysts, and potentially the public for informational purposes.  
- **Inputs Required:**  
  - Father’s and mother’s monthly incomes  
  - Children’s ages (up to 4)  
  - Exception flags (if applicable)  
- **Output:** A **range** (lower–upper bound) of possible monthly maintenance.  

⚠️ **Not legal advice.** Actual court orders may differ due to case-specific factors not captured by the model (e.g., special expenses, caregiving arrangements).

---

## Training Data
- **Source:** 10 spreadsheets of cases (2020–2023)  
- **Stage 1 cleaning:** 46 cases retained from ~500 (strict numeric completeness + outlier filtering)  
- **Stage 2 cleaning:** Improved parsing/exception handling → **69 cases retained**  
- **Final dataset used:** 69 clean cases + ML-synthetic augmentation  
- **Excluded data:** Case 01715-2023 (text formatting prevented numeric parsing; can be reinstated with manual cleaning)  

---

## Data Cleaning & Preparation
- **Required fields:** maintenance awarded, father income, mother income, number of children, ages.  
- **Blanks:** treated as `NaN`, not zero.  
- **Eligibility:**  
  - `<21 years` → eligible  
  - `≥21 years` → included only if schooling, in NS, or disabled  
- **Outlier removal:**  
  - Parent income > $20,000  
  - Award > $4,000/month  
  - Award ÷ combined income > 80%  

Result: a **trustworthy, fully numeric dataset** for modelling.

---

## Synthetic Data Strategy
- **Early stage (46 cases):** Augmented with both **ML synthetic** and **AI synthetic (TVAE-like)**.  
- **Later stage (69 cases):** Stronger real dataset → only **ML synthetic** retained.  
- **Weighting:** Real cases weighted 3× in training; synthetic down-weighted to prevent drift.  
- **Outcome:** Wider reliance on synthetic early, reduced once more real cases were available.  

---

## Model Development & Evolution
- **Phase 1 (46 + synthetic):**  
  - Tried linear regression, Elastic Net, Random Forest, Gradient Boosting.  
  - **Random Forest** performed best initially, but produced **wide ranges**.  

- **Phase 2 (69 + ML synthetic):**  
  - Re-ran models with expanded dataset.  
  - **Switched to a per-child calibrated model**, delivering narrower, realistic ranges.  

- **Final model:**  
  - **Per-child calibrated banded model**  
  - Outputs rounded ranges (nearest $10)  
  - Implemented in `CMPerChildModelRounded`  

---

## Calibration, Banding & Rounding
- **Problem:** Early models gave overly precise ranges (e.g., $254–$366), which felt unrealistic.  
- **Solution:**  
  - Band predictions into **policy-consistent intervals**  
  - Round both ends of range to nearest **$10** inside the model (not just UI)  
- **Result:**  
  - Human-friendly ranges, improved user trust  
  - Consistency across app, exports, and reviews  

---

## Evaluation & Validation
- **Validation method:** Blind expert review  
  - Spreadsheet with model outputs, drop-downs for expert judgment (*Within Range / Over / Under*)  
  - Comments field for qualitative feedback  
  - True awards hidden from reviewers (held by DLA as answer key)  
- **Planned metrics:**  
  - Capture rate (% actual within predicted band)  
  - Band width (sharpness vs coverage)  
  - Expert feedback on anomalies  

---

## Ethical Considerations & Limitations
- **Data size:** Only 69 clean real cases; small dataset limits statistical generalisation.  
- **Synthetic augmentation:** Improves coverage but carries risk of synthetic drift.  
- **Case scope:**  
  - Designed for typical 1–4 child cases.  
  - Predictions for >4 children are less reliable due to limited training data.  
- **Biases:** Reflects cases available in LAB spreadsheets (2020–2023); may not generalise to all socio-economic contexts.  
- **Disclaimer:** Provides **estimates only** — not binding or prescriptive.  

---

## Deployment
- **Platform:** Streamlit app (`app.py`)  
- **Inputs:** incomes, children’s ages (up to 4), exception flags  
- **Outputs:** range of maintenance (rounded to nearest $10)  
- **Notes:**  
  - 4 child cap is both a **UI simplification** and reflects **data scarcity** beyond 4 children  
  - Future versions may extend but with wider ranges/disclaimer  

---

## Maintenance & Versioning
- **Model version:** `v2 calibrated banded` (August 2025)  
- **Files:**  
  - `model_per_child_v2_calibrated_banded.joblib`  
  - `CMPerChildModelRounded` class wrapper  
- **Version control:** Each retrain should increment model version and update card.  
- **Future retraining triggers:**  
  - Arrival of new real case data  
  - Expert validation feedback  
  - Change in legal framework (e.g., maintenance guidelines)  

---
