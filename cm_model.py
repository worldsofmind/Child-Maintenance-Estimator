import numpy as np
import pandas as pd

class _BasePredictorMixin:
    """Shared predict() and snapping helpers."""
    def __init__(self):
        self.prep = None
        self.gb_pc_ = None
        self.iso_ = None
        self.round_outputs = True

    def predict(self, X: pd.DataFrame):
        # Preprocess
        Zt = self.prep.transform(X) if getattr(self, "prep", None) is not None else X
        # Per-child base prediction
        y_pc = self.gb_pc_.predict(Zt)
        # Optional isotonic calibration
        if getattr(self, "iso_", None) is not None:
            try:
                y_pc = self.iso_.predict(y_pc)
            except Exception:
                pass
        # Total = per-child * Eligible_Child_Count (if present)
        if "Eligible_Child_Count" in X.columns:
            total = y_pc * X["Eligible_Child_Count"].to_numpy()
        else:
            total = y_pc
        if getattr(self, "round_outputs", True):
            total = np.rint(total).astype(int)
        return total

    # ---------- Rounding policy helpers ----------
    @staticmethod
    def _snap_centered_cap_vectorized(point: np.ndarray, base: int = 50, cap: float = 200.0):
        """Centered fixed-width snapping:
        - interval center = point
        - width = `cap`
        - endpoints rounded to nearest `base`
        Guarantees hi > lo, clamps lo >= 0.
        """
        lo = np.round((point - cap/2.0) / base) * base
        hi = np.round((point + cap/2.0) / base) * base
        same = hi <= lo
        hi[same] = lo[same] + base
        lo = np.maximum(lo, 0)
        return lo.astype(int), hi.astype(int)

    @staticmethod
    def _snap_outward_vectorized(lo: np.ndarray, hi: np.ndarray, base: int = 50):
        """Outward snapping to a grid: floor lower to base; ceil upper to base; clamp lo>=0."""
        lo_r = (np.floor(lo / base) * base).astype(int)
        hi_r = (np.ceil(hi / base) * base).astype(int)
        lo_r = np.maximum(lo_r, 0)
        return lo_r, hi_r


class CMPerChildModelRounded(_BasePredictorMixin):
    """
    Original container class (artifact-friendly).
    Exposes:
      - predict(X) -> total monthly integer
      - predict_interval(X) -> lawyer-friendly DISPLAY interval (default: centered cap=$200, base=$50)
      - predict_interval_raw_banded(X) -> legacy banded interval from artifact's band_halfwidths_
    Expected attributes (in artifact):
      - prep: sklearn transformer
      - gb_pc_: regressor predicting per-child
      - iso_: optional isotonic calibrator
      - band_halfwidths_: dict {'low','mid','high'} (legacy raw banding)
      - q_low, q_mid: floats, band thresholds (default 0.4, 0.8)
    """
    def __init__(self):
        super().__init__()
        self.band_halfwidths_ = None
        self.q_low = 0.4
        self.q_mid = 0.8

    # Legacy raw band as reference (not used for display)
    def predict_interval_raw_banded(self, X: pd.DataFrame):
        point = self.predict(X).astype(float)
        if not isinstance(getattr(self, "band_halfwidths_", None), dict):
            raise AttributeError("band_halfwidths_ not found on model; cannot compute raw banded interval.")
        inc = (X["Father_income_cleaned"] + X["Mother_income_cleaned"]).to_numpy()
        q_low = getattr(self, "q_low", 0.4)
        q_mid = getattr(self, "q_mid", 0.8)
        t_low = np.quantile(inc, q_low)
        t_mid = np.quantile(inc, q_mid)
        bands = np.where(inc <= t_low, "low", np.where(inc <= t_mid, "mid", "high"))
        hw = np.array([self.band_halfwidths_[b] for b in bands], dtype=float)
        lo = np.maximum(point - hw, 0)
        hi = point + hw
        return np.rint(lo).astype(int), np.rint(hi).astype(int)

    # Display default: centered cap 200, snapped to $50
    def predict_interval(self, X: pd.DataFrame, base: int = 50, cap: float = 200.0):
        point = self.predict(X).astype(float)
        lo, hi = self._snap_centered_cap_vectorized(point, base=base, cap=cap)
        return lo, hi


class CMModelHybridTight(_BasePredictorMixin):
    """
    Hybrid tight raw interval:
      halfwidth = max(abs85_band, rel85_band * point), clamped to [floor_abs, cap_abs]
      bands via combined-income thresholds t_low (40%), t_mid (80%).
    Display default: centered cap=$200, base=$50.
    """
    def __init__(self, t_low, t_mid, abs85, rel85, floor_abs=25.0, cap_abs=200.0):
        super().__init__()
        self.t_low = float(t_low); self.t_mid = float(t_mid)
        self.abs85 = dict(abs85)
        self.rel85 = {k: float(v) for k,v in rel85.items()}
        self.floor_abs = float(floor_abs); self.cap_abs = float(cap_abs)

    def _band(self, inc):
        return "low" if inc <= self.t_low else ("mid" if inc <= self.t_mid else "high")

    def predict_interval_raw(self, X: pd.DataFrame):
        point = self.predict(X).astype(float)
        inc = (X["Father_income_cleaned"] + X["Mother_income_cleaned"]).to_numpy()
        bands = np.array([self._band(v) for v in inc])
        w_abs = np.array([self.abs85[b] for b in bands], dtype=float)
        w_rel = np.array([self.rel85[b] for b in bands], dtype=float) * point
        w = np.maximum(w_abs, w_rel)
        w = np.clip(w, self.floor_abs, self.cap_abs)
        lo = np.maximum(point - w, 0)
        hi = point + w
        return np.rint(lo).astype(int), np.rint(hi).astype(int)

    def predict_interval(self, X: pd.DataFrame, base: int = 50, cap: float = 200.0):
        point = self.predict(X).astype(float)
        return self._snap_centered_cap_vectorized(point, base=base, cap=cap)


class CMModelCrossConformal(_BasePredictorMixin):
    """
    Cross-conformal style raw interval:
      width = c(quartile) where quartiles defined by q25,q50,q75; fallback to c_global.
    Display default: centered cap=$200, base=$50.
    """
    def __init__(self, q25, q50, q75, c_by_quart, c_global):
        super().__init__()
        self.q25 = float(q25); self.q50 = float(q50); self.q75 = float(q75)
        self.c_by_quart = dict(c_by_quart); self.c_global = float(c_global)

    def _quart(self, inc):
        if inc <= self.q25: return "Q1"
        elif inc <= self.q50: return "Q2"
        elif inc <= self.q75: return "Q3"
        else: return "Q4"

    def predict_interval_raw(self, X: pd.DataFrame):
        point = self.predict(X).astype(float)
        inc = (X["Father_income_cleaned"] + X["Mother_income_cleaned"]).to_numpy()
        qu = np.array([self._quart(v) for v in inc])
        c = np.array([self.c_by_quart.get(q, self.c_global) for q in qu], dtype=float)
        lo = np.maximum(point - c, 0); hi = point + c
        return np.rint(lo).astype(int), np.rint(hi).astype(int)

    def predict_interval(self, X: pd.DataFrame, base: int = 50, cap: float = 200.0):
        point = self.predict(X).astype(float)
        return self._snap_centered_cap_vectorized(point, base=base, cap=cap)
