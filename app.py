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
try:
    import numpy.random._pickle as _np_random_pickle  # type: ignore[attr-defined]
    _orig_randomstate_ctor = getattr(_np_random_pickle, "__randomstate_ctor", None)
    if _orig_randomstate_ctor is not None:
        def _patched_randomstate_ctor(*args, **kwargs):
            try:
                return _orig_randomstate_ctor(*args, **kwargs)
            except TypeError:
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
    pos = fobj.tell()
    try:
        return joblib.load(fobj)
    except Exception as e_joblib:
        try: fobj.seek(pos)
        except Exception: pass
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

with st.sidebar:
    st.header("Options")
    show_point = st.checkbox("Show point prediction", value=True)
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
    exc = st.selectbox("Exception case (NS/schooling/disability)", options=[0, 1], index=0, key="exc")

with c2:
    st.markdown("**Children's Ages**")
    ages = []
    for i in range(1, 5):
        if i <= int(child_count):
            r1, r2 = st.columns([1, 2])
            u = r1.checkbox(f"Child {i} under 1", value=False, key=f"u{i}")
            if u:
                r2.markdown("Age: **Under 1** (≈ 6 months)")
                ages.append(0.5)
            else:
                yrs = r2.number_input(f"Child {i} age (years)", min_value=0.0, max_value=25.0,
                                      step=1.0, value=0.0, format="%.0f", key=f"a{i}_years")
                ages.append(yrs)
        else:
            ages.append(0.0)

go = st.button("Predict")

if go:
    child_count = int(child_count)
    exc = int(exc)
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
        width = 200
        lo = int(max(0, y_pred - width // 2))
        hi = int(y_pred + width // 2)
        def _snap50(v): return int(round(v / 50.0) * 50)
        lo, hi = _snap50(lo), _snap50(hi)

    st.subheader("Predicted monthly child maintenance")
    if show_point:
        st.info(f"Point estimate: **${y_pred:,}**")
    if lo is not None and hi is not None:
        st.success(f"Range: **${lo:,} — ${hi:,}**")

# Footer
try:
    import sklearn, numpy, pandas
    st.caption(f"Env → sklearn {sklearn.__version__}, numpy {numpy.__version__}, pandas {pandas.__version__}")
except Exception:
    pass
