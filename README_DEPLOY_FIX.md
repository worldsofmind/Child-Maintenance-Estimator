# Streamlit Deploy Fix Files

This bundle contains the minimal changes to make your app load the joblib model reliably on Streamlit Cloud.

## Files
- `app.py` — patched `_load_model_from_repo` that loads with `joblib.load(path)` first, then falls back.
- `requirements.txt` — bumps `numpy` to 1.24.0 and `joblib` to 1.5.1 (fixes RandomState unpickling issue).
- `runtime.txt` — pins Python to 3.11 on Streamlit Cloud.

## How to use
1. Replace these files at the **repo root** (same folder as your current `app.py`).
2. Commit and push to GitHub.
3. In Streamlit Cloud: **Manage app → Reboot and clear cache** (or push a new commit to trigger a fresh build).

After redeploy, your model should load without the `__randomstate_ctor` error.
