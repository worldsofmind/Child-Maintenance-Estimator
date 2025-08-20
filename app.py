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
            # Newer NumPy expects 0-1 args; some old pickles pass 2.
            try:
                return _orig_randomstate_ctor(*args, **kwargs)
            except TypeError:
                # Fall back to using only the first positional argument (state)
                if len(args) >= 1:
                    return _orig_randomstate_ctor(args[0])
                raise
        _np_random_pickle.__randomstate_ctor = _patched_randomstate_ctor  # type: ignore
except Exception:
    # Best-effort; if anything fails we leave defaults
    pass

# Ensure custom class is resolvable even if pickled under __main__
import sys as _sys
try:
    from cm_model import CMPerChildModelRounded as _CMCls
    import __main__ as _mn
    _mn.CMPerChildModelRounded = _CMCls
except Exception:
    pass
# --- End compatibility patch ---


# Ensure custom classes are importable for unpickling
from cm_model import CMPerChildModelRounded

st.set_page_config(page_title="Child Maintenance — Calculator", page_icon="👶", layout="centered")

CANDIDATE_FILENAMES = [
    "model_per_child_v2_calibrated_banded_rounded.joblib",
    "model_hybrid_tight.joblib",
    "model_cross_conformal.joblib",
]

def try_load_joblib(fobj):
    try:
        return joblib.load(fobj)
    except Exception as e_joblib:
        try:
            return cloudpickle.load(fobj)
        except Exception as e_cp:
            raise RuntimeError(f"Failed to load model via joblib ({e_joblib}) and cloudpickle ({e_cp})")

def load_local_model():
    here = Path(__file__).parent.resolve()
    for name in CANDIDATE_FILENAMES:
        p = here / name
        if p.exists():
            with open(p, "rb") as f:
                return try_load_joblib(f)
    return None

def compute_eligible_count(ages, exception_case: int) -> int:
    ages_nn = [a for a in ages if a is not None]
    under21 = [a for a in ages_nn if a < 21]
    adults  = [a for a in ages_nn if a >= 21]
    return len(under21) + (1 if (int(exception_case)==1 and len(adults)>0) else 0)

st.title("Child Maintenance — Prediction + Range")
st.caption("Outputs show a **$200-wide range rounded to the nearest $50** (lawyer-friendly display).")

with st.sidebar:
    st.markdown("### Model")
    uploaded = st.file_uploader("Upload model (.joblib)", type=["joblib","pkl","pkl.gz"], accept_multiple_files=False)
    model = None
    if uploaded is not None:
        try:
            model = try_load_joblib(BytesIO(uploaded.read()))
            st.success("Loaded uploaded model.")
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
    if 'model' not in locals() or model is None:
        model = load_local_model()

    if model is None:
        st.error("No model available. Upload a .joblib in the sidebar or place it next to app.py.")
    else:
        ages = [a1, a2 if child_count>=2 else None, a3 if child_count>=3 else None, a4 if child_count>=4 else None]
        ages = [float(a) for a in ages if a is not None]
        ages = sorted(ages, reverse=True)[:4]
        ages += [None]*(4-len(ages))

        eligible_count = compute_eligible_count(ages, int(exc))

        row = {
            "Father_income_cleaned": float(father),
            "Mother_income_cleaned": float(mother),
            "No. of children of the marriage": int(child_count),
            "Child1_Age": ages[0], "Child2_Age": ages[1], "Child3_Age": ages[2], "Child4_Age": ages[3],
            "exception_case": int(exc),
            "Eligible_Child_Count": int(eligible_count),
        }
        X_df = pd.DataFrame([row])

        try:
            y_pred = model.predict(X_df)
        except Exception as e:
            st.error(f"Model predict failed: {e}")
            st.stop()

        y_pred = int(np.array(y_pred).ravel()[0])

        lo, hi = None, None
        if hasattr(model, "predict_interval"):
            try:
                lo_arr, hi_arr = model.predict_interval(X_df)
                lo, hi = int(np.array(lo_arr).ravel()[0]), int(np.array(hi_arr).ravel()[0])
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
                st.write("Ages (oldest→youngest):", [ages[0], ages[1], ages[2], ages[3]])