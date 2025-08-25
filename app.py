import streamlit as st
import pandas as pd, numpy as np, joblib, cloudpickle
from io import BytesIO
from pathlib import Path

# --- Compatibility patch for numpy RandomState pickles (handles __randomstate_ctor signature mismatches) ---
try:
    import numpy.random._pickle as _np_random_pickle  # type: ignore
    _orig_randomstate_ctor = getattr(_np_random_pickle, "__randomstate_ctor", None)
    if _orig_randomstate_ctor is not None:
        def _patched_randomstate_ctor(*args, **kwargs):
            try:
                return _orig_randomstate_ctor(*args, **kwargs)
            except TypeError:
                if len(args) >= 1:
                    return _orig_randomstate_ctor(args[0])
                raise
        _np_random_pickle.__randomstate_ctor = _patched_randomstate_ctor  # type: ignore
except Exception:
    pass

# Ensure custom class is resolvable even if pickled under __main__
try:
    from cm_model import CMPerChildModelRounded as _CMCls
    import __main__ as _mn
    _mn.CMPerChildModelRounded = _CMCls
except Exception:
    pass

# ----------------- Helpers -----------------
def try_load_joblib(fobj: BytesIO):
    pos = fobj.tell()
    try:
        return joblib.load(fobj)
    except Exception as e_joblib:
        try: fobj.seek(pos)
        except Exception: pass
        try:
            return cloudpickle.load(fobj)
        except Exception as e_cp:
            raise RuntimeError(f"Failed to load model via joblib ({e_joblib}) and cloudpickle ({e_cp})")

def load_local_model():
    here = Path(__file__).parent.resolve()
    for name in [
        "model_per_child_v2_calibrated_banded_rounded.joblib",
        "model_hybrid_tight.joblib",
        "model_cross_conformal.joblib",
    ]:
        p = here / name
        if p.exists():
            with open(p, "rb") as f:
                return try_load_joblib(f)
    return None

def compute_eligible_count(ages, exception_case: int) -> int:
    # ages: list of floats (0 means N/A)
    ages = [a for a in ages if a is not None]
    eligible = 0
    for a in ages:
        if a <= 0:
            continue
        if a < 21:
            eligible += 1
        else:
            if int(exception_case) == 1:
                eligible += 1
    return eligible

def build_feature_row(father, mother, child_count, ages, exception_case):
    # ages as list length 4, floats
    a1, a2, a3, a4 = [float(x) for x in (ages + [0,0,0,0])[:4]]
    # Base columns expected by the model
    d = {
        "Father_income_cleaned": float(father),
        "Mother_income_cleaned": float(mother),
        "No. of children of the marriage": int(child_count),
        "Child1_Age": a1,
        "Child2_Age": a2,
        "Child3_Age": a3,
        "Child4_Age": a4,
        "exception_case": int(exception_case),
    }
    # Eligible count
    eligible_count = compute_eligible_count([a1,a2,a3,a4], exception_case)
    d["Eligible_Child_Count"] = int(eligible_count)

    # Derived income features
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

    # Age arrays (0 means N/A)
    ages_all = np.array([a1, a2, a3, a4], dtype=float)
    ages_all = np.where(ages_all <= 0, np.nan, ages_all)

    # 'All children' age features
    d["Youngest_Age_All"] = float(np.nanmin(ages_all)) if not np.isnan(ages_all).all() else 0.0
    d["Oldest_Age_All"]   = float(np.nanmax(ages_all)) if not np.isnan(ages_all).all() else 0.0
    d["Avg_Age_All"]      = float(np.nanmean(ages_all)) if not np.isnan(ages_all).all() else 0.0
    d["Age_Gap_All"]      = d["Oldest_Age_All"] - d["Youngest_Age_All"]

    d["Count_Under7"]  = int(np.nansum(ages_all < 7))
    d["Count_Under12"] = int(np.nansum(ages_all < 12))
    d["Count_Under18"] = int(np.nansum(ages_all < 18))
    d["Has_Adult"]     = int(np.nanmax(ages_all) >= 18 if not np.isnan(ages_all).all() else 0)

    # Eligible-only ages
    ages_elig = ages_all.copy()
    if int(exception_case) == 0:
        ages_elig = np.where(ages_elig >= 21, np.nan, ages_elig)
    # else: keep 21+ as eligible

    d["Youngest_Age_Eligible"] = float(np.nanmin(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    d["Oldest_Age_Eligible"]   = float(np.nanmax(ages_elig)) if not np.isnan(ages_elig).all() else 0.0
    d["Avg_Age_Eligible"]      = float(np.nanmean(ages_elig)) if not np.isnan(ages_elig).all() else 0.0

    d["Eligible_Under12"]     = int(np.nansum(ages_elig < 12))
    d["Eligible_Under18"]     = int(np.nansum(ages_elig < 18))
    d["Has_Eligible_Adult"]   = int(np.nanmax(ages_elig) >= 18 if not np.isnan(ages_elig).all() else 0)

    d["No_Children"] = int(int(child_count) == 0)
    d["Children_to_Eligible_Ratio"] = (int(child_count) / max(eligible_count, 1)) if int(child_count) > 0 else 0.0

    # 1-row DataFrame with exact column names
    X = pd.DataFrame([d])
    return X, eligible_count

# ----------------- UI -----------------
st.set_page_config(page_title="Child Maintenance Estimator", page_icon="👶", layout="centered")
st.title("Child Maintenance Estimator")

# Sidebar: model loading
with st.sidebar:
    st.header("Model")
    uploaded = st.file_uploader("Upload .joblib (optional)", type=["joblib","pkl"])
    model = None
    if uploaded is not None:
        try:
            model = try_load_joblib(uploaded)
            st.success("Uploaded model loaded.")
        except Exception as e:
            st.error(f"Could not load uploaded model: {e}")
    if model is None:
        model = load_local_model()
        if model:
            st.success("Loaded model from repository.")
        else:
            st.warning("No model found. Upload a model or add it to the repo root.")

    st.markdown("### Options")
    show_point = st.checkbox("Show point prediction", value=True)
    show_details = st.checkbox("Show details", value=False)

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        father = st.number_input("Father income (monthly)", min_value=0.0, step=50.0, value=2000.0)
        child_count = st.number_input("No. of children of the marriage", min_value=1, max_value=4, step=1, value=1)
        exc = st.selectbox("Exception case (NS/schooling/disability)", options=[0,1], index=0)
    with c2:
        mother = st.number_input("Mother income (monthly)", min_value=0.0, step=50.0, value=1500.0)
        a1 = st.number_input("Child1 age (oldest)", min_value=0.0, max_value=25.0, step=1.0, value=10.0)
        a2 = st.number_input("Child2 age", min_value=0.0, max_value=25.0, step=1.0, value=0.0, help="Leave 0 if not applicable")
        a3 = st.number_input("Child3 age", min_value=0.0, max_value=25.0, step=1.0, value=0.0, help="Leave 0 if not applicable")
        a4 = st.number_input("Child4 age", min_value=0.0, max_value=25.0, step=1.0, value=0.0, help="Leave 0 if not applicable")
    go = st.form_submit_button("Predict")

if go:
    if model is None:
        st.error("No model available. Upload a .joblib in the sidebar or place it next to app.py.")
    else:
        ages = [a1, a2 if child_count>=2 else 0.0, a3 if child_count>=3 else 0.0, a4 if child_count>=4 else 0.0]
        X, eligible_count = build_feature_row(father, mother, child_count, ages, exc)

        try:
            y = model.predict(X)
            y_pred = int(float(np.atleast_1d(y)[0]))
        except Exception as e:
            st.error(f"Model predict failed: {e}")
            st.stop()

        lo, hi = None, None
        try:
            lo_arr, hi_arr = model.predict_interval(X)
            lo = int(float(np.atleast_1d(lo_arr)[0]))
            hi = int(float(np.atleast_1d(hi_arr)[0]))
        except Exception:
            pass

        st.subheader("Predicted monthly child maintenance")
        if show_point:
            st.info(f"Point estimate: **${y_pred:,}**")
        if lo is not None and hi is not None:
            st.success(f"Range: **${lo:,} — ${hi:,}** (fixed width $200, rounded to $50)")
        else:
            st.warning("Range unavailable for this artifact. Upload a compatible model.")

        if show_details:
            with st.expander("Details", expanded=False):
                st.write(f"Eligible children used: **{eligible_count}**")
                st.write("Ages (oldest→youngest):", [a1, a2, a3, a4])

# Show environment footer for debugging
try:
    import sklearn, numpy, pandas
    st.caption(f"Env → sklearn {sklearn.__version__}, numpy {numpy.__version__}, pandas {pandas.__version__}")
except Exception:
    pass
