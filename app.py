import streamlit as st
import pandas as pd, numpy as np, joblib, cloudpickle, hashlib, os, datetime
from io import BytesIO
from pathlib import Path

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Child Maintenance Estimator", page_icon="👶", layout="centered")

# -----------------------------------------------------------------------------
# Compatibility shims (safe, no-ops if not needed)
# -----------------------------------------------------------------------------
# NumPy RandomState pickle signature shim (handles "__randomstate_ctor" TypeError in some pickles)
try:
    import numpy.random._pickle as _np_random_pickle  # type: ignore[attr-defined]
    _orig_randomstate_ctor = getattr(_np_random_pickle, "__randomstate_ctor", None)
    if _orig_randomstate_ctor is not None:
        def _patched_randomstate_ctor(*args, **kwargs):
            try:
                return _orig_randomstate_ctor(*args, **kwargs)
            except TypeError:
                # Some old pickles pass 2 positional args; use just the first (state)
                if len(args) >= 1:
                    return _orig_randomstate_ctor(args[0])
                raise
        _np_random_pickle.__randomstate_ctor = _patched_randomstate_ctor  # type: ignore[attr-defined]
except Exception:
    pass

# Ensure the custom class is resolvable even if it was pickled under __main__
try:
    from cm_model import CMPerChildModelRounded as _CMCls
    import __main__ as _mn
    _mn.CMPerChildModelRounded = _CMCls
except Exception:
    pass

# -----------------------------------------------------------------------------
# Model loading (auto, cached). No uploader shown to end users.
# -----------------------------------------------------------------------------
MODEL_FILENAME = "model_per_child_v2_calibrated_banded_rounded.joblib"

def try_load_joblib(fobj: BytesIO):
    """Attempt joblib first, then cloudpickle (for rare cases)."""
    pos = fobj.tell()
    try:
        return joblib.load(fobj)
    except Exception as e_joblib:
        try:
            fobj.seek(pos)
        except Exception:
            pass
        try:
            return cloudpickle.load(fobj)
        except Exception as e_cp:
            raise RuntimeError(
                f"Failed to load model via joblib ({e_joblib}) and cloudpickle ({e_cp})"
            )

def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@st.cache_resource
def _load_model_from_repo(path_str: str, file_hash: str):
    """Cache key includes file hash so replacing the model auto-invalidates cache."""
    with open(path_str, "rb") as f:
        return try_load_joblib(f)

_here = Path(__file__).parent.resolve()
_model_path = _here / MODEL_FILENAME
if not _model_path.exists():
    st.error(f"Model file '{MODEL_FILENAME}' not found in the repo root. Add it next to app.py and redeploy.")
    st.stop()

_model_hash = _file_md5(_model_path)
model = _load_model_from_repo(str(_model_path), _model_hash)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def money(x: float) -> str:
    """Format currency without triggering math mode (escaped $, shown as S$)."""
    return f"S\\${int(x):,}"

# -----------------------------------------------------------------------------
# Feature engineering (build exactly what the model expects)
# -----------------------------------------------------------------------------
def compute_eligible_count(ages, exception_case: int) -> int:
    eligible = 0
    for a in ages:
        if a is None or a <= 0:
            continue
        if a < 21:
            eligible += 1
        else:
            if int(exception_case) == 1:
                eligible += 1
    return eligible

def build_feature_row(father, mother, child_count, ages, exception_case):
    a1, a2, a3, a4 = [float(x) for x in (ages + [0, 0, 0, 0])[:4]]
    d = {
        "Father_income_cleaned": float(father),
        "Mother_income_cleaned": float(mother),
        "No. of children of the marriage": int(child_count),
        "Child1_Age": a1, "Child2_Age": a2, "Child3_Age": a3, "Child4_Age": a4,
        "exception_case": int(exception_case),
    }
    eligible_count = compute_eligible_count([a1, a2, a3, a4], exception_case)
    d["Eligible_Child_Count"] = int(eligible_count)

    combined = d["Father_income_cleaned"] + d["Mother_income_cleaned"]
    d["Combined_Income"] = combined
    d["Income_Diff_Abs"] = abs(d["Father_income_cleaned"] - d["Mother_income_cleaned"])
    d["Father_Share"] = (d["Father_income_cleaned"] / combined) if combined != 0 else 0.0
    d["Mother_Share"] = (d["Mother_income_cleaned"] / combined) if combined != 0 else 0.0
    den = max(eligible_count, 1)
    d["Combined_Income_per_Eligible"] = combined / den
    d["Father_Income_per_Eligible"] = d["Father_income_cleaned"] / den
    d["Mother_Income_per_Eligible"] = d["Mother_income_cleaned"] / den
    d["Is_Single_Income"] = int(d["Father_income_cleaned"] == 0 or d["Mother_income_cleaned"] == 0)
    d["Combined_Income_Zero"] = int(combined == 0)

    ages_all = np.array([a1, a2, a3, a4], dtype=float)
    ages_all = np.where(ages_all <= 0, np.nan, ages_all)

    d["Youngest_Age_All"] = float(np.nanmin(ages_all)) if not np.isnan(ages_all).all() else 0.0
    d["Oldest_Age_All"]   = float(np.nanmax(ages_all)) if not np.isnan(ages_all).all() else 0.0
    d["Avg_Age_All"]      = float(np.nanmean(ages_all)) if not np.isnan(ages_all).all() else 0.0
    d["Age_Gap_All"]      = d["Oldest_Age_All"] - d["Youngest_Age_All"]

    d["Count_Under7"]  = int(np.nansum(ages_all < 7))
    d["Count_Under12"] = int(np.nansum(ages_all < 12))
    d["Count_Under18"] = int(np.nansum(ages_all < 18))
    d["Has_Adult"]     = int(np.nanmax(ages_all) >= 18 if not np.isnan(ages_all).all() else 0)

    ages_elig = ages_all.copy()
    if int(exception_case) == 0:
        ages_elig = np.where(ages_elig >= 21, np.nan, ages_elig)

    d["Youngest_Age_Eligible"] = float(np.nanmin(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    d["Oldest_Age_Eligible"]   = float(np.nanmax(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    d["Avg_Age_Eligible"]      = float(np.nanmean(ages_elig)) if not np.isnan(ages_elig).all() else 0.0

    d["Eligible_Under12"]   = int(np.nansum(ages_elig < 12))
    d["Eligible_Under18"]   = int(np.nansum(ages_elig < 18))
    d["Has_Eligible_Adult"] = int(np.nanmax(ages_elig) >= 18 if not np.isnan(ages_elig).all() else 0)

    d["No_Children"] = int(int(child_count) == 0)
    d["Children_to_Eligible_Ratio"] = (int(child_count) / max(eligible_count, 1)) if int(child_count) > 0 else 0.0

    return pd.DataFrame([d]), eligible_count

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Child Maintenance Estimator")

# Compact description
st.info(
    "**What this tool does**  \n"
    "• Gives a quick, ballpark estimate of the **family’s total monthly child maintenance**.  \n"
    "• Built for **practitioners** (e.g., legal clinics); **not** public self-service.  \n"
    "• **Supports up to 4 children** today; future updates will allow more.  \n"
)

with st.sidebar:
    st.header("Options")
    show_point = st.checkbox("Show point prediction", value=False)
    # tiny model/env status footer
    try:
        import sklearn, numpy, pandas
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(_model_path))
        st.caption(
            f"Model loaded • {MODEL_FILENAME} • updated {ts:%Y-%m-%d %H:%M} • "
            f"sklearn {sklearn.__version__}, numpy {numpy.__version__}, pandas {pandas.__version__}"
        )
    except Exception:
        pass

# Inputs (no form → widgets rerender instantly)
c1, c2 = st.columns(2)

with c1:
    father = st.number_input("Father income (monthly)", min_value=0.0, step=50.0, value=0.0,
                             format="%.0f", key="father_income")
    mother = st.number_input("Mother income (monthly)", min_value=0.0, step=50.0, value=0.0,
                             format="%.0f", key="mother_income")
    child_count = st.number_input("No. of children of the marriage", min_value=1.0, max_value=4.0,
                                  step=1.0, value=1.0, format="%.0f", key="child_count")
    exc_choice = st.radio(
        "Do any children aged 21 or older still qualify as dependent (NS / still studying full-time / disability)?",
        ["No", "Yes"], horizontal=True,
        help="Select 'Yes' if at least one child aged 21+ is still dependent due to National Service, still studying full-time, or disability."
    )
    exc = 1 if exc_choice == "Yes" else 0

with c2:
    st.markdown("**Children's Ages**")
    ages = []
    for i in range(1, 5):
        if i <= int(child_count):
            r1, r2 = st.columns([1, 2])
            u = r1.checkbox(
                f"Child {i} under 1 year",
                value=False,
                key=f"u{i}",
            )
            if u:
                r2.caption("Counted as less than 12 months")
                ages.append(0.5)  # internal assumption
            else:
                yrs = r2.number_input(
                    f"Child {i} age (years)",
                    min_value=0.0, max_value=25.0, step=1.0, value=0.0, format="%.0f",
                    key=f"a{i}_years"
                )
                ages.append(yrs)
        else:
            ages.append(0.0)

# Context hint: if any age ≥ 21 while exceptions = No, warn the user
try:
    has_over21 = any(a >= 21 for a in ages[:int(child_count)])
    if has_over21 and exc == 0:
        st.warning("You entered an age of 21 or older. If that child is still dependent (NS/still studying full-time/disability), switch the option above to **Yes** to count them as eligible.")
except Exception:
    pass

go = st.button("Predict")

if go:
    # Coerce numeric types
    child_count = int(child_count)

    # If both incomes are zero: show warning, stop (no numbers)
    combined_income = float(father) + float(mother)
    if combined_income == 0.0:
        st.warning(
            "Cannot generate an estimate when both parents' incomes are S$0. "
            "Please enter at least one parent's income to continue."
        )
        st.stop()

    X, eligible_count = build_feature_row(father, mother, child_count, ages, exc)

    # Predict
    try:
        y = model.predict(X)
        y_pred = int(float(np.atleast_1d(y)[0]))
    except Exception as e:
        st.error(f"Model predict failed: {e}")
        st.stop()

    # Interval (prefer model's own; otherwise display-only fallback: centered $200 width, $50 rounding)
    lo, hi = None, None
    try:
        lo_arr, hi_arr = model.predict_interval(X)
        lo = int(float(np.atleast_1d(lo_arr)[0]))
        hi = int(float(np.atleast_1d(hi_arr)[0]))
    except Exception:
        width = 200
        lo = int(max(0, y_pred - width // 2))
        hi = int(y_pred + width // 2)
        def _snap50(v): return int(round(v / 50.0) * 50)
        lo, hi = _snap50(lo), _snap50(hi)

    # Output (escape $ to avoid math font)
    st.subheader("Predicted monthly child maintenance")
    if show_point:
        st.info(f"Point estimate: **{money(y_pred)}**")
    if lo is not None and hi is not None:
        st.success(f"Range: **{money(lo)} — {money(hi)}**")
