#!/usr/bin/env python3
"""
test_api.py — Test script for the Olist Review Prediction API.

Usage:
    python test_api.py                                # test localhost:5000
    python test_api.py https://your-app.onrender.com  # test deployed API
"""

import sys
import json
import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:5000"

# ── Shared valid sample (mirrors model/sample_input.json structure) ────────────
VALID_SAMPLE = {
    "delivery_days": 12.0,
    "delivery_vs_estimated": 3.0,
    "price": 149.99,
    "freight_value": 25.50,
    "seller_score": 4.2,
    "num_previous_sales": 10.0,
    "cust_reviews": 4.5,
    "freight_ratio": 0.17,
    "num_previous_reviews": 5,
    "num_items": 1,
    "same_state": 1,
    "is_repeat_customer": 0,
    "delivery_missing": 0,
    "product_category_name_english": "electronics",
    "seller_state": "SP",
    "payment_type": "credit_card",
}


# ── Test runner helpers ────────────────────────────────────────────────────────
passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print("="*60)
    try:
        fn()
        print(f"  ✓  PASSED")
        passed += 1
    except AssertionError as e:
        print(f"  ✗  FAILED — {e}")
        failed += 1


# ── Test 1: Health check ───────────────────────────────────────────────────────
def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=60)
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {json.dumps(r.json(), indent=4)}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "status" in data,  "Response missing 'status' field"
    assert "model"  in data,  "Response missing 'model' field"
    assert data["status"] == "healthy", f"Expected 'healthy', got '{data['status']}'"
    assert data["model"]  == "loaded",  f"Expected 'loaded', got '{data['model']}'"


# ── Test 2: Valid single prediction ───────────────────────────────────────────
def test_single_prediction():
    r = requests.post(f"{BASE_URL}/predict", json=VALID_SAMPLE, timeout=60)
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {json.dumps(r.json(), indent=4)}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "prediction"  in data, "Response missing 'prediction' field"
    assert "probability" in data, "Response missing 'probability' field"
    assert "label"       in data, "Response missing 'label' field"
    assert data["prediction"] in [0, 1],          "prediction must be 0 or 1"
    assert 0.0 <= data["probability"] <= 1.0,     "probability must be in [0, 1]"
    assert data["label"] in ["positive", "negative"], "label must be 'positive' or 'negative'"


# ── Test 3: Valid batch of 5 records ──────────────────────────────────────────
def test_batch_prediction():
    batch = [VALID_SAMPLE.copy() for _ in range(5)]
    # Vary a couple of fields so the records aren't identical
    batch[1]["price"] = 49.99
    batch[2]["delivery_days"] = 5.0
    batch[3]["payment_type"] = "boleto"
    batch[4]["seller_state"] = "RJ"

    r = requests.post(f"{BASE_URL}/predict/batch", json=batch, timeout=60)
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {json.dumps(r.json(), indent=4)}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "predictions" in data,   "Response missing 'predictions' field"
    assert "count"       in data,   "Response missing 'count' field"
    assert data["count"] == 5,      f"Expected 5 predictions, got {data['count']}"
    assert len(data["predictions"]) == 5, "predictions array length != 5"
    for pred in data["predictions"]:
        assert "prediction"  in pred
        assert "probability" in pred
        assert "label"       in pred


# ── Test 4: Missing required field → 400 ──────────────────────────────────────
def test_missing_field():
    # Send only two fields — 14 required fields are missing
    incomplete = {"delivery_days": 12.0, "price": 149.99}

    r = requests.post(f"{BASE_URL}/predict", json=incomplete, timeout=60)
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {json.dumps(r.json(), indent=4)}")

    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    data = r.json()
    assert "error"   in data,             "400 response missing 'error' field"
    assert "details" in data,             "400 response missing 'details' field"
    assert "missing_fields" in data["details"], \
        "details should list 'missing_fields'"


# ── Test 5: Invalid type (string for a numeric field) → 400 ───────────────────
def test_invalid_type():
    bad = VALID_SAMPLE.copy()
    bad["price"] = "not-a-number"   # price must be numeric

    r = requests.post(f"{BASE_URL}/predict", json=bad, timeout=60)
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {json.dumps(r.json(), indent=4)}")

    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    data = r.json()
    assert "error"   in data, "400 response missing 'error' field"
    assert "details" in data, "400 response missing 'details' field"
    assert "price"   in data["details"], \
        "details should identify 'price' as the invalid field"


# ── Run all tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nTesting API at: {BASE_URL}\n")

    run_test("Health Check (GET /health)",                   test_health)
    run_test("Valid Single Prediction (POST /predict)",       test_single_prediction)
    run_test("Valid Batch of 5 (POST /predict/batch)",        test_batch_prediction)
    run_test("Missing Required Field → 400",                  test_missing_field)
    run_test("Invalid Type for 'price' → 400",               test_invalid_type)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("="*60)

    sys.exit(0 if failed == 0 else 1)