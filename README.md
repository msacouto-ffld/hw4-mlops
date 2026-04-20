# Olist Customer Satisfaction Prediction API

A REST API that predicts whether an Olist customer will leave a positive review, based on order and delivery features. The model is a proactive early-warning system — predictions are made before the review is written, using signals like delivery time, freight cost, seller history, and customer behavior.

## Live URL

**https://msacouto-hw4-mlops.onrender.com**

---

## API Documentation

### `GET /health`

Confirms the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "model_type": "Pipeline",
  "n_features": 16
}
```

---

### `POST /predict`

Single prediction. Send a JSON object with order features, receive a prediction with probability.

**Request body:**
```json
{
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
  "payment_type": "credit_card"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.8836,
  "label": "positive"
}
```

**Error response (400):**
```json
{
  "error": "Invalid input",
  "details": {
    "price": "expected a number, got 'str'"
  }
}
```

---

### `POST /predict/batch`

Batch prediction. Send a JSON array of up to 100 records, receive predictions for all.

**Request body:** JSON array of objects, each following the same schema as `/predict`.

**Response:**
```json
{
  "predictions": [
    {"prediction": 1, "probability": 0.8836, "label": "positive"},
    {"prediction": 0, "probability": 0.3102, "label": "negative"}
  ],
  "count": 2
}
```

---

## Input Schema

| Feature | Type | Valid Values / Range | Nullable |
|---|---|---|---|
| `delivery_days` | float | ≥ 0 | No |
| `delivery_vs_estimated` | float | Any (negative = early delivery) | No |
| `price` | float | ≥ 0 | No |
| `freight_value` | float | ≥ 0 | No |
| `seller_score` | float | 1–5 (avg of past seller reviews) | Yes — null for new sellers |
| `num_previous_sales` | float | ≥ 0 | Yes |
| `cust_reviews` | float | 1–5 (avg of past customer reviews) | Yes — null for new customers |
| `num_previous_reviews` | int | ≥ 0 | Yes |
| `num_items` | int | ≥ 1 | No |
| `freight_ratio` | float | ≥ 0 | No |
| `same_state` | int | 0 or 1 | No |
| `is_repeat_customer` | int | 0 or 1 | No |
| `delivery_missing` | int | 0 or 1 | No |
| `product_category_name_english` | string | See allowed values below | No |
| `seller_state` | string | Brazilian state codes (AC, AM, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RS, SC, SE, SP) | No |
| `payment_type` | string | `boleto`, `credit_card`, `debit_card`, `not_defined`, `voucher` | No |

<details>
<summary>Allowed values for <code>product_category_name_english</code> (71 categories)</summary>

`agro_industry_and_commerce`, `air_conditioning`, `art`, `arts_and_craftmanship`, `audio`, `auto`, `baby`, `bed_bath_table`, `books_general_interest`, `books_imported`, `books_technical`, `cds_dvds_musicals`, `christmas_supplies`, `cine_photo`, `computers`, `computers_accessories`, `consoles_games`, `construction_tools_construction`, `construction_tools_lights`, `construction_tools_safety`, `cool_stuff`, `costruction_tools_garden`, `costruction_tools_tools`, `diapers_and_hygiene`, `drinks`, `dvds_blu_ray`, `electronics`, `fashio_female_clothing`, `fashion_bags_accessories`, `fashion_childrens_clothes`, `fashion_male_clothing`, `fashion_shoes`, `fashion_sport`, `fashion_underwear_beach`, `fixed_telephony`, `flowers`, `food`, `food_drink`, `furniture_bedroom`, `furniture_decor`, `furniture_living_room`, `furniture_mattress_and_upholstery`, `garden_tools`, `health_beauty`, `home_appliances`, `home_appliances_2`, `home_comfort_2`, `home_confort`, `home_construction`, `housewares`, `industry_commerce_and_business`, `kitchen_dining_laundry_garden_furniture`, `la_cuisine`, `luggage_accessories`, `market_place`, `music`, `musical_instruments`, `office_furniture`, `party_supplies`, `perfumery`, `pet_shop`, `security_and_services`, `signaling_and_security`, `small_appliances`, `small_appliances_home_oven_and_coffee`, `sports_leisure`, `stationery`, `tablets_printing_image`, `telephony`, `toys`, `watches_gifts`

</details>

---

## Local Setup

### Without Docker

```bash
# 1. Clone the repo
git clone https://github.com/msacouto-ffld/hw4-mlops.git
cd hw4-mlops

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
python app.py
# API available at http://localhost:5000

# 5. Run tests (in a second terminal)
python test_api.py
```

### With Docker

```bash
# 1. Build the image
docker build -t hw4-api .

# 2. Run the container
docker run -p 5000:5000 hw4-api
# API available at http://localhost:5000

# 3. Run tests (in a second terminal)
python test_api.py
```

### Testing against the live Render deployment

```bash
python test_api.py https://msacouto-hw4-mlops.onrender.com
```

---

## Model Information

| | |
|---|---|
| **Model** | `HistGradientBoostingClassifier` (scikit-learn) |
| **Pipeline** | Preprocessing (median imputation + standard scaling for numerics, most-frequent imputation + one-hot encoding for categoricals) + classifier, serialized as a single `joblib` object |
| **Training data** | Olist Brazilian e-commerce dataset (~117k orders) |
| **Train/test split** | 80/20, stratified, `random_state=42` |
| **Key hyperparameters** | `learning_rate=0.05`, `max_depth=5`, `max_iter=300`, `class_weight='balanced'` |
| **Accuracy** | 0.817 |
| **F1 Score** | 0.879 |
| **ROC-AUC** | 0.827 |

**Known limitations:**

- The model relies heavily on customer and seller review history. For first-time customers or new sellers, `cust_reviews` and `seller_score` are null and fall back to training medians via imputation, which may reduce prediction quality.
- `delivery_missing` is a training-data artifact flagging orders with no recorded delivery date. In production this field should be set to `1` only when delivery data is genuinely unavailable.
- The model was trained on Olist data from 2016–2018. Significant shifts in delivery norms or category mix may reduce accuracy over time (see `part5_monitoring.ipynb` for drift analysis).