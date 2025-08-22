
import streamlit as st
import pandas as pd, numpy as np, joblib, cloudpickle, hashlib, os, datetime
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="Child Maintenance Estimator", page_icon="👶", layout="centered", initial_sidebar_state="collapsed")

# Compatibility: ensure custom class resolvable
try:
    from cm_model import CMPerChildModelRounded as _CMCls
    import __main__ as _mn
    _mn.CMPerChildModelRounded = _CMCls
except Exception:
    pass

CANDIDATE_MODEL_FILENAMES = [
    "gb_per_child_perchild_symmetric_STRICT.joblib",
    "gb_per_child_perchild_symmetric.joblib",
    "model_per_child_v2_calibrated_banded_rounded.joblib"
]

def try_load_joblib(fobj: BytesIO):
    pos = fobj.tell()
    try:
        return joblib.load(fobj)
    except Exception as e_joblib:
        try: fobj.seek(pos)
        except Exception: pass
        try: return cloudpickle.load(fobj)
        except Exception as e_cp:
            raise RuntimeError(f"Failed to load model via joblib ({e_joblib}) and cloudpickle ({e_cp})")

def _file_md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@st.cache_resource
def _load_model_from_repo(path_str: str, file_hash: str):
    with open(path_str, "rb") as f:
        return try_load_joblib(f)

_here = Path(__file__).parent.resolve()
_model_path = None
for nm in CANDIDATE_MODEL_FILENAMES:
    p = _here / nm
    if p.exists():
        _model_path = p
        break

if _model_path is None:
    st.error("No model file found. Expected one of: " + ", ".join(CANDIDATE_MODEL_FILENAMES))
    st.stop()

_model_hash = _file_md5(_model_path)
model = _load_model_from_repo(str(_model_path), _model_hash)

# Wrap per-child pipeline if needed
try:
    from cm_model import CMPerChildModelRounded as _CMCls
    if not hasattr(model, "predict_interval"):
        _wrapper = _CMCls()
        _wrapper.prep = None
        _wrapper.gb_pc_ = model
        _wrapper.iso_ = None
        model = _wrapper
except Exception:
    pass

def money(x: float) -> str:
    return f"S\\${int(x):,}"

# Strict symmetric features (no raw father/mother columns)
def compute_eligible_count(ages, exception_case: int) -> int:
    eligible = 0
    for a in ages:
        if a is None or a <= 0: continue
        if a < 21: eligible += 1
        else:
            if int(exception_case) == 1: eligible += 1
    return eligible

STRICT_FEATURE_ORDER = ["No. of children of the marriage", "Child1_Age", "Child2_Age", "Child3_Age", "Child4_Age", "exception_case", "Eligible_Child_Count", "Combined_Income", "Income_Diff_Abs", "Income_Min", "Income_Max", "Combined_Income_per_Eligible", "Youngest_Age_All", "Oldest_Age_All", "Avg_Age_All", "Age_Gap_All", "Count_Under7", "Count_Under12", "Count_Under18", "Youngest_Age_Eligible", "Oldest_Age_Eligible", "Avg_Age_Eligible", "Eligible_Under12", "Eligible_Under18", "Has_Eligible_Adult"]

def build_feature_row(father, mother, child_count, ages, exception_case):
    a1, a2, a3, a4 = [float(x) for x in (ages + [0,0,0,0])[:4]]
    elig = compute_eligible_count([a1,a2,a3,a4], exception_case)

    combined = float(father) + float(mother)
    income_diff_abs = abs(float(father) - float(mother))
    income_min = min(float(father), float(mother))
    income_max = max(float(father), float(mother))
    den = max(elig, 1)
    combined_per_elig = combined / den

    ages_all = np.array([a1,a2,a3,a4], dtype=float)
    ages_all = np.where(ages_all <= 0, np.nan, ages_all)

    youngest_all = float(np.nanmin(ages_all)) if not np.isnan(ages_all).all() else 0.0
    oldest_all   = float(np.nanmax(ages_all)) if not np.isnan(ages_all).all() else 0.0
    avg_all      = float(np.nanmean(ages_all)) if not np.isnan(ages_all).all() else 0.0
    gap_all      = oldest_all - youngest_all

    cnt_u7  = int(np.nansum(ages_all < 7))
    cnt_u12 = int(np.nansum(ages_all < 12))
    cnt_u18 = int(np.nansum(ages_all < 18))

    ages_elig = ages_all.copy()
    if int(exception_case) == 0:
        ages_elig = np.where(ages_elig >= 21, np.nan, ages_elig)
    youngest_elig = float(np.nanmin(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    oldest_elig   = float(np.nanmax(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    avg_elig      = float(np.nanmean(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    elig_u12      = int(np.nansum(ages_elig < 12)) if not np.isnan(ages_elig).all() else 0
    elig_u18      = int(np.nansum(ages_elig < 18)) if not np.isnan(ages_elig).all() else 0
    has_elig_adult = int(np.nanmax(ages_elig) >= 18 if not np.isnan(ages_elig).all() else 0)

    row = {
        "No. of children of the marriage": int(child_count),
        "Child1_Age": a1, "Child2_Age": a2, "Child3_Age": a3, "Child4_Age": a4,
        "exception_case": int(exception_case),
        "Eligible_Child_Count": int(elig),
        "Combined_Income": combined,
        "Income_Diff_Abs": income_diff_abs,
        "Income_Min": income_min,
        "Income_Max": income_max,
        "Combined_Income_per_Eligible": combined_per_elig,
        "Youngest_Age_All": youngest_all,
        "Oldest_Age_All": oldest_all,
        "Avg_Age_All": avg_all,
        "Age_Gap_All": gap_all,
        "Count_Under7": cnt_u7,
        "Count_Under12": cnt_u12,
        "Count_Under18": cnt_u18,
        "Youngest_Age_Eligible": youngest_elig,
        "Oldest_Age_Eligible": oldest_elig,
        "Avg_Age_Eligible": avg_elig,
        "Eligible_Under12": elig_u12,
        "Eligible_Under18": elig_u18,
        "Has_Eligible_Adult": has_elig_adult,
    }
    X = pd.DataFrame([row])
    # Reorder to match training exactly
    X = X[STRICT_FEATURE_ORDER]
    return X, int(elig)

st.title("Child Maintenance Estimator")
st.info("**What this tool does**  \n• Gives a quick, ballpark estimate of the **family’s total monthly child maintenance**.  \n• Built for **practitioners** (e.g., legal clinics); **not** public self-service.  \n• **Supports up to 4 children** today; future updates will allow more.")

with st.expander("Learn more about this tool", expanded=False):
    st.markdown("""
**Purpose**  
Gives a quick, realistic starting point for the **family’s total monthly child maintenance**. Built for **practitioners** (e.g., legal clinics), not public self-service.

**How it works**  
Trained on LAB actual case data.

**Disclaimer**  
The predicted maintenance range is an estimate based on provided inputs and should not be considered as legal or financial advice.
""")

with st.sidebar:
    st.header("Options")
    show_point = st.checkbox("Show point prediction", value=False)
    try:
        import sklearn, numpy, pandas
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(_model_path))
        st.caption(f"Model loaded • {_model_path.name} • updated {ts:%Y-%m-%d %H:%M} • sklearn {sklearn.__version__}, numpy {numpy.__version__}, pandas {pandas.__version__}")
    except Exception:
        pass

c1, c2 = st.columns(2)

with c1:
    father = st.number_input("Father income (monthly)", min_value=0.0, step=50.0, value=0.0, format="%.0f", key="father_income")
    mother = st.number_input("Mother income (monthly)", min_value=0.0, step=50.0, value=0.0, format="%.0f", key="mother_income")
    child_count = st.number_input("No. of children of the marriage", min_value=1.0, max_value=4.0, step=1.0, value=1.0, format="%.0f", key="child_count")
    exc_choice = st.radio("Do any children aged 21 or older still qualify as dependent (NS / still studying full-time / disability)?", ["No", "Yes"], horizontal=True)
    exc = 1 if exc_choice == "Yes" else 0

with c2:
    st.markdown("**Children's Ages**")
    ages = []
    for i in range(1, 5):
        if i <= int(child_count):
            r1, r2 = st.columns([1, 2])
            u = r1.checkbox(f"Child {i} under 1 year", value=False, key=f"u{i}")
            if u:
                r2.caption("Counted as less than 12 months")
                ages.append(0.5)
            else:
                yrs = r2.number_input(f"Child {i} age (years)", min_value=0.0, max_value=25.0, step=1.0, value=0.0, format="%.0f", key=f"a{i}_years")
                ages.append(yrs)
        else:
            ages.append(0.0)

try:
    has_over21 = any(a >= 21 for a in ages[:int(child_count)])
    if has_over21 and exc == 0:
        st.warning("You entered an age of 21 or older. If that child is still dependent (NS/still studying full-time/disability), switch the option above to **Yes** to count them as eligible.")
except Exception:
    pass

go = st.button("Predict")

if go:
    child_count = int(child_count)

    combined_income = float(father) + float(mother)
    if combined_income == 0.0:
        st.warning("Cannot generate an estimate when both parents' incomes are S$0. Please enter at least one parent's income to continue.")
        st.stop()

    X, eligible_count = build_feature_row(father, mother, child_count, ages, exc)

    try:
        y = model.predict(X)
        y_pred = int(float(np.atleast_1d(y)[0]))
    except Exception as e:
        st.error(f"Model predict failed: {e}")
        st.stop()

    try:
        lo_arr, hi_arr = model.predict_interval(X)
        lo = int(float(np.atleast_1d(lo_arr)[0])); hi = int(float(np.atleast_1d(hi_arr)[0]))
    except Exception:
        width = 200
        lo = int(max(0, y_pred - width // 2))
        hi = int(y_pred + width // 2)
        def _snap50(v): return int(round(v / 50.0) * 50)
        lo, hi = _snap50(lo), _snap50(hi)

    st.subheader("Predicted monthly child maintenance")
    if show_point:
        st.info(f"Point estimate: **{money(y_pred)}**")
    st.success(f"Range: **{money(lo)} — {money(hi)}**")
