# Child Maintenance Model — Technical Write-up

## Goal
Predict **monthly total child maintenance** and provide a credible **prediction interval**, using raw user inputs (incomes, number of children, ages, exception flag).

## Data
Source workbook: `For model building CM_no_outliers_incl_exceptions_with_eligible_count.xlsx`

Key columns:
- Father/Mother income: `Father_income_cleaned`, `Mother_income_cleaned`
- Children: `No. of children of the marriage`, `Child1_Age`…`Child4_Age`
- Flags: `exception_case` (NS/schooling/disability)
- Target: `Total_child_maintenance_awarded`
- Derived: `Eligible_Child_Count`

### Eligibility rule
```
Eligible_Child_Count = (# children < 21) + 1 * [exception_case == 1 AND >=1 child >= 21]
```

## Formulation
- Train a **per-child** regressor (`gb_pc_`) → multiply by `Eligible_Child_Count` to get **total**.
- Optional **isotonic calibrator** (`iso_`) for monotonic, well-calibrated outputs.
- Round to whole dollars.

## Serving wrapper
`CMPerChildModelRounded` (in `cm_model.py`) exposes:
- `predict(X)` → total monthly integer
- `predict_interval(X)` → (lower, upper) using income-banded halfwidths

## Features
- Raw: incomes, #children, 4 ages, `exception_case`, `Eligible_Child_Count`
- Engineered: combined income, income shares/diff, per-eligible scaling, age summary stats (all & eligible-only), flags.
- **Preprocessor:** ColumnTransformer with 34 numeric features..

## Algorithm
- Base model: GradientBoostingRegressor (params below)
- Params:
```
{
  "alpha": 0.9,
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "squared_error",
  "max_depth": 3,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```
- Calibration: Isotonic present
- Intervals (artifact): income-banded with halfwidths:
```
{
  "low": 104.22115384615383,
  "mid": 94.79166666666667,
  "high": 101.06666666666659
}
```

## Evaluation
_Evaluation workbook not found in this environment._

## How to run
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn joblib cloudpickle
streamlit run app.py
```
Upload the model in the sidebar or place it in the same folder as `app.py`.
