import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ============================================================
# Initialize app and load artifacts
# ============================================================
app = Flask(__name__)

# Load model ONCE at startup — not per request (key lesson from Demo 1A)
MODEL_PATH  = os.environ.get("MODEL_PATH",  "model/model.pkl")
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", "model/schema.json")

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(model).__name__}")

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

REQUIRED_FEATURES    = schema["all_features"]
NUMERIC_CONSTRAINTS  = schema["numeric_features"]
CATEGORICAL_ALLOWED  = schema["categorical_features"]

BATCH_LIMIT = 100


# ============================================================
# Helper: validate a single input dict
# Returns (is_valid: bool, errors: dict)
# ============================================================
def validate_input(data):
    errors = {}

    # 1. Missing fields
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        errors["missing_fields"] = missing
        return False, errors  # no point checking further if fields are absent

    # 2. Numeric fields — type check + range/constraint check
    for field, rules in NUMERIC_CONSTRAINTS.items():
        value = data[field]

        # Null check
        if value is None:
            if not rules["nullable"]:
                errors[field] = "field is required and cannot be null"
            continue  # null + nullable → skip range checks

        # Type check
        try:
            value = float(value)
        except (ValueError, TypeError):
            errors[field] = f"expected a number, got '{type(data[field]).__name__}'"
            continue

        # Range checks
        if rules.get("min") is not None and value < rules["min"]:
            errors[field] = f"must be >= {rules['min']}, got {value}"
        if rules.get("max") is not None and value > rules["max"]:
            errors[field] = f"must be <= {rules['max']}, got {value}"

    # 3. Categorical fields — type check + allowed-value check
    for field, rules in CATEGORICAL_ALLOWED.items():
        value = data[field]

        if value is None:
            errors[field] = "field is required and cannot be null"
            continue

        if not isinstance(value, str):
            errors[field] = f"expected a string, got '{type(value).__name__}'"
            continue

        if value not in rules["allowed"]:
            errors[field] = (
                f"unrecognized value '{value}'. "
                f"Allowed values: {rules['allowed']}"
            )

    return len(errors) == 0, errors


# ============================================================
# Endpoints
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check — confirms API is running and model is loaded."""
    return jsonify({
        "status":     "healthy",
        "model":      "loaded",
        "model_type": type(model).__name__,
        "n_features": len(REQUIRED_FEATURES),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Single prediction — POST a JSON object, get prediction + probability."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    is_valid, errors = validate_input(data)
    if not is_valid:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    # Build DataFrame in the exact column order the model expects
    df = pd.DataFrame([data])[REQUIRED_FEATURES]

    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    label       = "positive" if prediction == 1 else "negative"

    return jsonify({
        "prediction":  prediction,
        "probability": round(probability, 4),
        "label":       label,
    }), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction — POST a JSON array (max 100 records)."""
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array"}), 400

    if len(data) == 0:
        return jsonify({"error": "Array must contain at least one record"}), 400

    if len(data) > BATCH_LIMIT:
        return jsonify({
            "error": f"Batch size {len(data)} exceeds the limit of {BATCH_LIMIT}"
        }), 400

    # Validate every record; collect all errors before returning
    all_errors = {}
    for i, record in enumerate(data):
        is_valid, errors = validate_input(record)
        if not is_valid:
            all_errors[f"record_{i}"] = errors

    if all_errors:
        return jsonify({"error": "Invalid input in batch", "details": all_errors}), 400

    # Build DataFrame and predict in one vectorised call
    df           = pd.DataFrame(data)[REQUIRED_FEATURES]
    predictions  = model.predict(df).tolist()
    probabilities = model.predict_proba(df)[:, 1].tolist()

    results = [
        {
            "prediction":  int(pred),
            "probability": round(float(prob), 4),
            "label":       "positive" if pred == 1 else "negative",
        }
        for pred, prob in zip(predictions, probabilities)
    ]

    return jsonify({
        "predictions": results,
        "count":       len(results),
    }), 200


# ============================================================
# Run (dev only — production uses gunicorn)
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)