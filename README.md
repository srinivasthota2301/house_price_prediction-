# Aetherline Valuation — King County House Price Prediction

Premium residential valuation platform trained on King County, Washington home sales. The project separates **offline model training** from a **Streamlit inference and analytics application**.

## Dataset

Source file: `data/kc_house_data.csv`

- 21,613 raw transactions, 21 columns
- Target: `price`
- Market window: May 2014 – May 2015
- `id` is excluded from model features
- `date` is parsed into sale year and sale month
- Repeat sales of the same `id` are retained as separate dated events
- Neighborhood fields `sqft_living15` and `sqft_lot15` are not used (unavailable on the valuation form)

Known quality issues handled in training:

- 33-bedroom record recoded to 3 bedrooms (inconsistent with living area)
- Zero-bedroom / zero-bathroom rows removed
- Feature engineering is applied identically for training and inference

## Architecture

```
house_price_prediction/
├── data/kc_house_data.csv
├── models/best_model.joblib
├── code/feature_engineering.py   # clean, engineer, train, evaluate, serialize
├── code/app.py                   # Streamlit product UI (no training)
├── README.md
└── requirements.txt
```

The Joblib bundle stores the sklearn pipeline **and** evaluation artifacts (metrics, holdout predictions, permutation importance, conformal interval quantile). The app never fits a model at startup.

## Installation

```bash
cd house_price_prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execution

Train and serialize the best pipeline:

```bash
python code/feature_engineering.py
```

Launch the product UI:

```bash
streamlit run code/app.py
```

## Machine learning methodology

1. **Cleaning** — validate schema, parse dates, repair the 33-bedroom typo, drop uninhabitable zero-count homes.
2. **Features** — house age, renovation flag, years since renovation, living-to-lot ratio, basement percentage, bathrooms per bedroom, living area per floor, sale year/month, plus original structural and location fields.
3. **Target** — log1p(price) via `TransformedTargetRegressor`, metrics reported in USD.
4. **Split** — chronological 80/20 by sale date (forward generalization). TimeSeriesSplit (3 folds) inside the training window for hyperparameter search.
5. **Models**
   - Ridge regression on log price (linear baseline)
   - Random Forest (nonlinear bagging)
   - XGBoost (justified: strong interactions among grade, living area, waterfront, and location)
6. **Selection** — lowest chronological holdout MAE. RMSE, R², and MAPE are reported. MAPE is treated as secondary because it over-weights inexpensive homes.

   Chronological holdout (March–May 2015, 4,320 homes):

   | Model | Test MAE | Test RMSE | Test R² | Test MAPE |
   | --- | --- | --- | --- | --- |
   | Ridge (log-price) | $82,676 | $142,877 | 0.849 | 14.24% |
   | Random Forest | $81,039 | $149,164 | 0.836 | 13.58% |
   | **XGBoost (selected)** | **$71,505** | **$126,543** | **0.882** | **12.56%** |
7. **Intervals** — 90% split-conformal band from absolute log residuals on the chronological test set.
8. **Explainability** — permutation importance (global) and median-input sensitivity (local). Neither is causal.

Valuation dates in the app default to the last month observed in training so `house_age` and `sale_year` stay inside the learned market window.

## Limitations

- Estimates reflect 2014–2015 King County transaction prices, not current list prices.
- Luxury tails have larger absolute errors.
- Intervals are empirical, not legal appraisal ranges.
- Location is represented by zipcode, latitude, and longitude, not school or amenity data.

## License / use

Educational and portfolio use on the public King County house sales dataset.
