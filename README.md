# housing-price-predictor
This project uses the California Housing dataset to build an end‑to‑end machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and performance evaluation. The goal is to understand the key drivers of housing prices and develop a reliable predictive model.
Project Structure:
├── data/                # Dataset
├── notebooks/           # Jupyter Notebooks for analysis
├── src/                 # Python scripts for modeling and preprocessing
├── README.md            # Project documentation


Exploratory Data Analysis (EDA):
Geographic distribution of housing prices
Correlation heatmap
Relationships between price and key variables (median_income, house_age, households)
Impact of ocean proximity

Data Preprocessing:
Missing value inspection
Outlier detection
Feature binning (e.g., age_bins, income_bins)
Categorical encoding (OneHotEncoder )
Train–test split

A variety of regression models were trained and compared:

Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
XGBoost Regressor
Hyperparameter tuning with GridSearchCV / RandomizedSearchCV

Model Performance:
Model	RMSE	R2
Linear Regression	68622.5353	0.6475
Ridge Regression	68622.5353	0.6476
Lasso Regression	68622.5353	0.6475
Random Forest Regressor	69457.9275	0.8172
XGBoost Regressor	49176.9580	0.8383

Key Insights:
Median income is the strongest predictor of housing prices
Coastal areas show significantly higher price levels
Housing age has a weaker relationship with price than expected
High‑density areas exhibit more price variability
XGBoost captures nonlinear patterns effectively and delivers the best performance
