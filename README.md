# NYC Taxi Fare Prediction (Regression Analysis)

This project builds an end-to-end machine learning pipeline to predict taxi `fare_amount` based on trip characteristics from the NYC Taxi Fare dataset.

The workflow covers the entire data science process:
1.  **Data Cleaning:** Loading, cleaning, and filtering a large dataset (6.4+ million rows).
2.  **Outlier Removal:** Implementing a statistical (IQR) method to remove outliers.
3.  **Feature Engineering:** Extracting temporal features from `datetime` columns.
4.  **Exploratory Data Analysis (EDA):** Visualizing data distributions, correlations, and relationships.
5.  **Modeling:** Training and comparing four different regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting).
6.  **Evaluation:** Assessing model performance using R², RMSE, MAE, and MAPE.

---

## 1. Data Pipeline

### a. Data Source
* **Dataset:** `dataFare.csv`
* **Raw Size:** The initial dataset contains **6,405,008** rows and 18 columns.

### b. Data Cleaning & Preprocessing
The raw data was extensively cleaned to ensure model quality:
* `dropna()` was used to remove rows with missing values.
* Logical filters were applied to remove nonsensical data (e..g, `trip_distance > 0`, `passenger_count > 0`, `fare_amount >= 0`).
* **Outlier Removal:** A robust outlier detection function using the **1.5 * IQR (Interquartile Range)** rule was applied to all numerical columns, significantly reducing data skewness.

### c. Feature Engineering
* `tpep_pickup_datetime` and `tpep_dropoff_datetime` were converted to `datetime` objects.
* Temporal features (`pickup_year`, `pickup_month`, `pickup_day`, `pickup_hour`) were extracted to capture time-based patterns.
* `store_and_fwd_flag` was mapped from categorical ('N'/'Y') to numerical (0/1).

### d. Exploratory Data Analysis (EDA)
A comprehensive EDA was performed on the cleaned data (`df_clean`) to understand feature relationships:
* **Correlation Heatmap:** Visualized the correlation between all numerical features.
* **Boxplots:** Compared the distribution of `fare_amount` *before* and *after* outlier removal.
* **Histograms:** Analyzed the distributions of key variables like `fare_amount`, `trip_distance`, and `tip_amount`.
* **Scatter Plots:** Investigated the relationships between `fare_amount` and other key drivers like `trip_distance` and `total_amount`.

---

## 2. Modeling & Evaluation

### a. Feature Selection & Pipeline
* **Target (Y):** `fare_amount`
* **Features (X):** `['trip_distance', 'tip_amount', 'total_amount']`
* **Pipeline:** A `ColumnTransformer` with `StandardScaler` was used to scale the features before training. The data was split 80% (Train) / 20% (Test).

### b. Model Comparison
Four models were trained, and their performance was evaluated on the unseen test set.

| Model | MAE | MSE | RMSE | R² (R-squared) | MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.3274** | **0.1419** | **0.3767** | **0.9886** | **4.18%** |
| Decision Tree | 0.3278 | 0.1445 | 0.3802 | 0.9884 | 4.19% |
| Gradient Boosting | 0.3362 | 0.1551 | 0.3939 | 0.9875 | 4.31% |
| Linear Regression | 0.3465 | 0.1506 | 0.3880 | 0.9879 | 4.55% |

---

## 3. Analysis & Conclusion

### a. Superficial Conclusion
Based purely on the metrics, the **Random Forest Regressor** was the best-performing model, achieving the highest R² (0.9886) and the lowest RMSE (0.3767). All models performed exceptionally well, with ~99% explained variance.

### b. CRITICAL ANALYSIS: Data Leakage
The near-perfect $R^2$ score ($\approx$ $0.99$) is a classic symptom of **Data Leakage**.

* **The Problem:** The model was trained to predict `fare_amount` (Y) using features (X) that included `tip_amount` and `total_amount`.
* **The Leak:** In this dataset, `total_amount` is a *result* of `fare_amount`. The relationship is:
    `total_amount = fare_amount + tip_amount + tolls_amount + ...`
* **Result:** The model did not *learn* to *predict* the fare; it simply *learned the formula* `fare_amount \approx total_amount - tip_amount`. This is why the predictions are almost perfect.

### c. Corrective Actions (Future Improvements)
To build a true *predictive* model, one must **only use features that are known *before* the fare is calculated**.

The model should be retrained with `total_amount`, `tip_amount`, `mta_tax`, and other fare components **removed** from the feature set (X).

A realistic feature set would be:
`X = ['trip_distance', 'passenger_count', 'pickup_hour', 'pickup_day', 'PULocationID', 'DOLocationID', ...]`

## 4. Technologies Used
* Python
* Pandas & NumPy
* Scikit-learn (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, StandardScaler, train_test_split)
* Matplotlib & Seaborn
