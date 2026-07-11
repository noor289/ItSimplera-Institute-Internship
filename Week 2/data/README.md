# Week 2 Internship Task — Steel Industry Energy Consumption
 
Deep exploratory analysis, feature engineering, and baseline regression modeling on the Steel Industry Energy Consumption dataset.
 
## Project Overview
 
This project has two parts:
 
1. **EDA & Feature Engineering** (`week2_eda.ipynb`) — explores the dataset, engineers new features from the raw columns, and documents data quality issues and patterns.
2. **Baseline Regression Modeling** (`week2_baseline_models.ipynb`) — uses the engineered features to train and compare four regression models for predicting `Usage_kWh`, and selects the best-performing one as a baseline.
## Dataset
 
- **Name:** Steel Industry Energy Consumption Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip)
- **Rows:** 35,040
- **Description:** Energy usage readings from a steel manufacturing plant, recorded every 15 minutes, including reactive power, power factor, CO2 emissions, load type, and timestamps.
- **Location in repo:** `data/Week_2_DataSet.xlsx`
## Environment Setup
 
```bash
# clone the repo
git clone <repo-url>
cd <repo-folder>
 
# create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
 
# install dependencies
pip install -r requirements.txt
 
# launch notebooks
jupyter notebook
```
 
## Feature Engineering Steps
 
Starting from the raw columns, the following features were added:
 
- `hour` — hour of day, extracted from `date`
- `month` — month, extracted from `date`
- `is_weekend` — binary flag, 1 if Saturday/Sunday
- `Power_Factor_Ratio` — `Leading_Current_Power_Factor / Lagging_Current_Power_Factor`, with 0/0 rows replaced by 1.0 (a balanced-state edge case rather than an undefined ratio)
- `High_Load` — binary flag, 1 if `Usage_kWh` is above the 75th percentile (used only in the EDA notebook; dropped before modeling since it directly leaks the target)
## EDA Findings
 
- No missing values across any column.
- `NSM` and `hour` are perfectly correlated (r = 1.00) — same information in different units, so `NSM` was dropped before modeling.
- `CO2(tCO2)` correlates at 0.99 with `Usage_kWh` because it is calculated directly from energy usage in this dataset, not measured independently. It was excluded from the correlation ranking and dropped before modeling to avoid target leakage.
- The genuinely informative correlated features are `Lagging_Current_Reactive.Power_kVarh` (0.90) and `Lagging_Current_Power_Factor` (0.39).
- IQR method flagged 328 outliers in `Usage_kWh`, consistent with a right-skewed distribution and genuine high-load periods rather than data errors.
- Usage follows a clear two-shift daily pattern: low overnight (5–8 kWh), sharp rise to 55–60 kWh by 8–9am, a midday dip to ~18 kWh, an afternoon recovery, and a drop-off after 8pm.
Full write-up is in the markdown summary cell at the end of `week2_eda.ipynb`.
 
## Model Training Process
 
1. Loaded the dataset with all engineered features from Part 1.
2. Dropped `date` (raw), `High_Load` (direct target leak), `CO2(tCO2)` (near-duplicate of target), and `NSM` (redundant with `hour`).
3. One-hot encoded `Load_Type`, `Day_of_week`, and `WeekStatus` — chosen over label encoding since these are nominal categories with no natural order, and label encoding would imply a false ranking (e.g. Wednesday > Monday).
4. Split 80/20 train/test with `random_state=42`.
5. Trained four models: Linear Regression, Ridge Regression, Decision Tree Regressor, Random Forest Regressor.
6. Evaluated each on MAE, RMSE, and R² on the test set, plus train-set metrics to check for overfitting.
7. Ran 5-fold cross-validation and reported mean RMSE per model.
## Results
 
| Model              | Train RMSE | Test RMSE | Train R² | Test R² | CV RMSE (mean) | CV RMSE (std) |
|---------------------|------------|-----------|----------|---------|-----------------|-----------------|
| Linear Regression   | 8.119      | 8.246     | 0.9408   | 0.9402  | 8.152           | 0.097           |
| Ridge Regression    | 8.119      | 8.247     | 0.9408   | 0.9402  | 8.152           | 0.097           |
| Decision Tree       | 0.000      | 1.345     | 1.0000   | 0.9984  | 1.427           | 0.059           |
| Random Forest       | 0.319      | 0.821     | 0.9999   | 0.9994  | 0.867           | 0.066           |
 
**Best model: Random Forest.**
 
- Linear and Ridge perform almost identically with no train/test gap — expected for a simple model with little capacity to overfit, but they underfit the non-linear relationship between the electrical features and `Usage_kWh`.
- Decision Tree shows clear overfitting: a perfect train R² of 1.0000 means it memorized the training data. Its test performance is still good, but the gap makes it less trustworthy on unseen data.
- Random Forest shows only a small train/test gap and stable, low-variance performance across CV folds, making it the most reliable model of the four.
Full metrics, charts, and the detailed model selection write-up are in `week2_baseline_models.ipynb`.
 
## Conclusions
 
Random Forest is carried forward as the baseline model, based on the lowest test and cross-validation RMSE, the highest R², and consistent performance across folds relative to the Decision Tree's overfitting. The strong performance of tree-based models over linear ones suggests the relationship between electrical readings (reactive power, power factor) and energy usage is meaningfully non-linear.
 
## Repository Structure
 
```
.
├── week2_eda.ipynb
├── week2_baseline_models.ipynb
├── data/
│   └── Week_2_DataSet.xlsx
├── README.md
└── requirements.txt
```
 
## Screenshots
 
**Usage_kWh Outlier Boxplot**
![Outlier Boxplot](images/outlier_boxplot.png)
 
**Correlation Heatmap**
![Correlation Heatmap](images/correlation_heatmap.png)

**Average Usage by Load Type**
![Load Type Usage](images/usage_load_type.png)
 
**Average Usage by Hour of Day**
![Hourly Usage](images/hourly_usage.png)
 
**Test RMSE Comparison Across Models**
![RMSE Comparison](images/rmse_comparison.png)
 
**Predicted vs Actual — Random Forest**
![Predicted vs Actual](images/predicted_vs_actual.png)
