import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


DATA_PATH  = "ml_ready_df.csv"   
MODEL_DIR  = "model"
TARGET_COL = "is_positive_review"
RANDOM_STATE = 42

# Best hyperparameters from HW2 GridSearchCV
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth":     5,
    "max_iter":      300,
}

# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ── 2. Feature lists ───────────────────────────────────────────────────────────

numeric_cols = [
    "delivery_days", "delivery_vs_estimated", "price", "freight_value",
    "seller_score", "num_previous_sales", "cust_reviews", "freight_ratio",
    "num_previous_reviews", "num_items",
    "same_state", "is_repeat_customer", "delivery_missing",  # binary ints — treated as numeric
]

cat_cols = [
    "product_category_name_english",
    "seller_state",
    "payment_type",
]

# ── 3. Train / test split (same as HW2 for reproducibility) ───────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. Build pipeline ──────────────────────────────────────

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", cat_pipeline,     cat_cols),
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        learning_rate=BEST_PARAMS["learning_rate"],
        max_depth=BEST_PARAMS["max_depth"],
        max_iter=BEST_PARAMS["max_iter"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])

# ── 5. Fit ─────────────────────────────────────────────────────────────────────

print("\nFitting pipeline...")
pipe.fit(X_train, y_train)

# ── 6. Sanity check ─────────────────────────

y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  F1       : {f1_score(y_test, y_pred):.3f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")

# ── 7. Save model.pkl ──────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(pipe, model_path)

size_mb = os.path.getsize(model_path) / (1024 ** 2)
print(f"\nSaved: {model_path}  ({size_mb:.1f} MB)")
assert size_mb < 100, "Model exceeds 100 MB — too large for free-tier deploy!"
loaded = joblib.load(model_path)
assert np.array_equal(loaded.predict(X_test.iloc[:5]), pipe.predict(X_test.iloc[:5])), \
    "Round-trip mismatch — model.pkl may be corrupt"
print("  Round-trip check")

# ── 8. Build and save schema.json ─────────────────────────────────────────────

# Collect unique values from training data for categorical validation
cat_allowed = {
    col: sorted(X_train[col].dropna().unique().tolist())
    for col in cat_cols
}

schema = {
    "numeric_features": {
        "delivery_days":          {"min": 0,    "max": None, "nullable": False},
        "delivery_vs_estimated":  {"min": None, "max": None, "nullable": False},  # negatives OK
        "price":                  {"min": 0,    "max": None, "nullable": False},
        "freight_value":          {"min": 0,    "max": None, "nullable": False},
        "seller_score":           {"min": 1,    "max": 5,    "nullable": True},   # NaN for new sellers
        "num_previous_sales":     {"min": 0,    "max": None, "nullable": True},
        "cust_reviews":           {"min": 1,    "max": 5,    "nullable": True},   # NaN for new customers
        "num_previous_reviews":   {"min": 0,    "max": None, "nullable": True},
        "num_items":              {"min": 1,    "max": None, "nullable": False},
        "freight_ratio":          {"min": 0,    "max": None, "nullable": False},
        "same_state":             {"min": 0,    "max": 1,    "nullable": False},
        "is_repeat_customer":     {"min": 0,    "max": 1,    "nullable": False},
        "delivery_missing":       {"min": 0,    "max": 1,    "nullable": False},
    },
    "categorical_features": {
        col: {"allowed": vals, "nullable": False}
        for col, vals in cat_allowed.items()
    },
    "all_features": numeric_cols + cat_cols,
}

schema_path = os.path.join(MODEL_DIR, "schema.json")
with open(schema_path, "w") as f:
    json.dump(schema, f, indent=2)
print(f"Saved: {schema_path}")

# ── 9. Save sample_input.json (first row from test set) ───────────────────────

sample_row = X_test.iloc[0].to_dict()

# Cast numpy types → Python types so json.dumps doesn't choke
def cast(v):
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
    return v

sample_row = {k: cast(v) for k, v in sample_row.items()}

sample_path = os.path.join(MODEL_DIR, "sample_input.json")
with open(sample_path, "w") as f:
    json.dump(sample_row, f, indent=2)

# Verify the sample produces a valid prediction
import pandas as pd
sample_df = pd.DataFrame([sample_row])
pred  = int(loaded.predict(sample_df)[0])
prob  = float(loaded.predict_proba(sample_df)[0][1])
print(f"Saved: {sample_path}")
print(f"  Sample prediction: {pred}  (probability: {prob:.4f})")
