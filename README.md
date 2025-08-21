# Child Maintenance Estimator

A Streamlit app that gives a quick, realistic **ballpark estimate of a family’s total monthly child maintenance**. It’s designed for **practitioners** (e.g., legal clinics), **not** public self-service.

> **Important:** This tool provides an indicative estimate only. It is **not** a court decision and **not** legal or financial advice.

---

## What this tool does / does not do

**Does**
- Estimates the **family’s total monthly maintenance** using a small set of high-signal inputs.
- Returns a **range** (kept reasonably tight for usability, about $200 wide in the pilot).
- Counts children **≥ 21** as eligible **only** if an exception applies (NS / still studying full-time / disability).
- **Supports up to 4 children** today; future updates will allow more.

**Does not**
- Decide the **payer split** (how much each parent pays).  
- Divide the total **per child** (by age/needs).  
- Replace practitioner judgement or detailed Women’s Charter factor analysis.

---

## Model overview

- **Training data:** LAB case data (pilot).  
- **Target:** Family **total** monthly child maintenance.  
- **Why a range?** Child maintenance is not an exact science; reasonable experts may differ. The range communicates uncertainty while staying usable for advice and negotiation.  
- **Eligibility rule (pilot):**  
  - All children **under 21** are included.  
  - If an **exception** applies (NS / still studying full-time / disability), include **up to one** adult child **≥ 21**. (Extendable later.)

---

## App UX summary

- **Auto-loaded model** (no uploader shown). The model file is cached and loaded from the repo.
- **Inputs start blank-ish** (zero incomes, zero ages).
- **Children’s ages**
  - Per-child **“Under 1”** toggle: if checked, the numeric age field **disappears** and the app uses **0.5 years** internally.
  - Age fields appear only for the selected **number of children** (1–4).
  - Ages are whole years (no decimals) for clarity.
- **Exception question** (clear phrasing):
  > “Do any children aged 21 or older still qualify as dependent (NS / still studying full-time / disability)?”
- **Over-21 hint:** If you enter any age ≥ 21 while the exception is **No**, the app shows a gentle warning to switch to **Yes** if appropriate.
- **Outputs**
  - **Point estimate** (optional toggle in sidebar).
  - **Range** (e.g., **$1,000 — $1,200**). Dollar signs are escaped in the app to avoid math-font glitches in Streamlit.

---

## Inputs & engineered features (key fields)

**User inputs**
- `Father income (monthly)`  
- `Mother income (monthly)`  
- `No. of children of the marriage` (1–4)  
- Per child: `Under 1` (checkbox) or `Age (years)`  
- `Exception` (NS / still studying full-time / disability): Yes/No

**Model features (non-exhaustive)**
- Income features: `Combined_Income`, `Income_Diff_Abs`, `Father_Share`, `Mother_Share`, per-eligible income variants, `Is_Single_Income`, `Combined_Income_Zero`.
- Age features across all/eligible children: youngest/oldest/average, `Age_Gap_All`, counts under 7/12/18, `Has_Adult`, eligible-only under 12/18, `Has_Eligible_Adult`.
- Counts: `Eligible_Child_Count`, `No_Children`, `Children_to_Eligible_Ratio`.
- Raw ages: `Child1_Age`–`Child4_Age`. (Ages of `0` mean “not applicable”.)

---

## Running the app

### 1) Place the model file
Put your model file in the repo **root**, next to `app.py`, with this exact name:
```
model_per_child_v2_calibrated_banded_rounded.joblib
```
> You can change the filename by updating `MODEL_FILENAME` in `app.py`.

### 2) Install & run (local)
```bash
python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate
pip install -r requirements.txt   # or: pip install streamlit pandas numpy scikit-learn joblib cloudpickle
streamlit run app.py
```

### 3) Deploy (Streamlit Community Cloud)
- Connect the GitHub repo.
- Ensure the model file is in the repo root.
- The app auto-loads the model and caches it.

---

## Updating the model

- **Drop a new file** (same filename) in the repo root and redeploy.  
  `app.py` computes an **MD5** of the model file and passes it to `st.cache_resource`, so changing the file **automatically invalidates** the cache and loads the new model.
- If you change the filename, update `MODEL_FILENAME` accordingly.

---

## Version compatibility (pickled models)

Pickled scikit-learn pipelines are **version-sensitive**. If the model was trained with:
- `scikit-learn==1.1.3`
- `numpy==1.23.5`
- `scipy==1.9.3`
- `joblib==1.2.0`

…you should **pin the same (or very close) versions in production**. Mismatches can cause errors like:
- `ModuleNotFoundError: sklearn.ensemble._gb_losses`
- `TypeError: __randomstate_ctor() takes from 0 to 1 positional arguments but 2 were given`
- `UnpicklingError` (cloudpickle)

This app includes **defensive shims** and a loader that tries both `joblib` and `cloudpickle`, but **best practice** is to align dependency versions with those used at training time.

---

## Common issues & fixes

- **Model file not found**  
  > “Model file '…joblib' not found…”  
  Ensure the file is in the repo root and the name matches `MODEL_FILENAME`.

- **Load failures (sklearn/numpy versions)**  
  Pin versions to the training environment in `requirements.txt`.

- **Weird font sizes in the output (e.g., “$100 — $300”)**  
  In Streamlit, a dollar sign can trigger math mode. The app **escapes** `$` to render normal text.

---

## Project layout (suggested)

```
.
├── app.py
├── model_per_child_v2_calibrated_banded_rounded.joblib
├── requirements.txt
└── README.md
```

---

## Roadmap

- Support **more than 4 children**.
- Add payer **split** (relative incomes) and **per-child** allocation views.
- More nuanced eligibility for multiple adult children.
- UX polish and printable PDF summary.

---

## Contributing

- Keep output strings user-friendly and concise.  
- Any change to feature engineering must match the training pipeline’s expected columns.

---

## License

Specify your project’s license here (e.g., MIT).
