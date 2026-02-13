# California Housing Price Prediction

A machine learning project that predicts California housing prices using census block-level data, comparing the performance of multiple regression models.

---

## Project Structure

```
├── CA_housing_price_prediction.ipynb   # Main analysis notebook
└── archive/
    └── housing.csv                     # Raw dataset
```

---

## Dataset

Uses the **California Housing Dataset** with **20,640** census block-level records:

| Feature | Description |
|---------|-------------|
| `longitude` / `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total number of rooms in the block |
| `total_bedrooms` | Total number of bedrooms (207 missing values) |
| `population` | Block population |
| `households` | Number of households in the block |
| `median_income` | Median household income |
| `ocean_proximity` | Distance to the ocean (categorical) |
| `median_house_value` | **Target variable**: Median house value |

---

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics (mean, std, quantiles)
- Distribution and box plot analysis of `median_house_value`
- Feature correlation heatmap
- Geographic scatter plot (longitude × latitude × house value × housing age)

### 2. Data Preprocessing
- Imputation of missing values in `total_bedrooms`
- Outlier detection and handling using the IQR method
- **One-Hot Encoding** of `ocean_proximity`
- Feature engineering: `age_bins` (binned housing age)
- Feature scaling with `StandardScaler`

### 3. Train/Test Split
- 80% training / 20% test, `random_state=42`

---

## Models & Results

All models are evaluated using **5-fold cross-validation**. Final test set performance is reported for the best model.

| Model | Avg R² (CV) | Avg RMSE (CV) |
|-------|:-----------:|:-------------:|
| Linear Regression | ~0.648 | ~$68,623 |
| Ridge (alpha=10) | ~0.648 | ~$68,623 |
| Lasso (alpha=32.37) | ~0.648 | ~$68,623 |
| Random Forest (100 trees) | ~0.818 | ~$49,266 |
| **XGBoost (tuned)** | **~0.836** | **~$49,266** |

**XGBoost final test set performance:**

| Metric | Value |
|--------|-------|
| R² | 0.8327 |
| RMSE | $46,815 |
| MSE | 2,191,674,302 |

### Hyperparameter Tuning
- **Ridge**: best `alpha=10` found via `GridSearchCV`
- **Lasso**: best `alpha=32.37` found via `LassoCV` over a log-spaced search grid
- **XGBoost**: best params `{n_estimators: 100, max_depth: 7, learning_rate: 0.1, colsample_bytree: 0.5}` found via `RandomizedSearchCV` (20 iterations, 5-fold CV)

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost
```

Recommended: Python 3.13+ with Anaconda.

---

## Getting Started

1. Place `housing.csv` in the `archive/` directory
2. Launch the notebook:
   ```bash
   jupyter notebook CA_housing_price_prediction.ipynb
   ```
3. Run cells sequentially from top to bottom

---

## Key Findings

- Linear models (Linear Regression, Ridge, Lasso) perform similarly at R² ≈ **0.648**, revealing a clear non-linear modeling ceiling.
- **Random Forest** significantly outperforms linear models with R² ≈ **0.818**, cutting RMSE nearly in half.
- **XGBoost** achieves the best overall performance at R² ≈ **0.836** (CV) and **0.833** on the held-out test set, with RMSE of ~$46,815.
- `median_income` is the strongest predictor of house value; geographic location (latitude/longitude) also plays an important role.
