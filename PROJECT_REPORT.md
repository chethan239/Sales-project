# MINI PROJECT REPORT
## Sales Forecasting & Predictive Analytics Using Machine Learning

### Abstract
This project develops a sales forecasting application using Python and Random Forest Regression. Users can upload a CSV file through a Streamlit dashboard or use the included sample data. The system prepares monthly sales data, engineers time-based features, evaluates predictions, and forecasts future sales.

### Problem Statement
To develop an interactive predictive analytics application that uses historical monthly sales data to estimate future sales and visualize trends and model performance.

### Objectives
1. Load historical sales data from CSV.
2. Allow users to upload their own dataset.
3. Clean and aggregate records into monthly sales.
4. Create lag and rolling-average features.
5. Train and evaluate a regression model.
6. Forecast future sales and display results in a dashboard.

### Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, Streamlit.

### Dataset
The sample dataset contains 60 generated monthly records with Date, Sales, Orders, and Marketing_Spend columns. It is synthetic demonstration data, not actual company data. For manual uploads, Date and Sales are required.

### Methodology
1. Read the CSV and validate required columns.
2. Convert Date and Sales to appropriate types and remove invalid rows.
3. Aggregate transaction rows by month.
4. Create month, year, lag-1, lag-2, lag-3, and rolling-three-month-mean features.
5. Train a Random Forest Regressor using earlier months.
6. Test on the final six chronological observations.
7. Calculate MAE, RMSE, and R².
8. Predict future months recursively and display the forecast.

### Algorithm
Random Forest Regression combines the outputs of multiple decision trees. For regression, the trees' predictions are averaged. It can model nonlinear patterns in historical sales data.

### Evaluation
- MAE: mean absolute difference between actual and predicted sales.
- RMSE: square root of mean squared error; it penalizes larger errors more.
- R²: indicates how much target variation is explained by the model.

The dashboard calculates actual metrics for the selected dataset. Do not invent or assume metric values.

### Expected Output
Historical sales chart, actual-versus-predicted chart, MAE/RMSE/R² metrics, future sales forecast table, and downloadable forecast CSV.

### Limitations
The sample data is synthetic. A small dataset, unusual events, and changing demand can reduce forecast quality. Recursive forecasts can accumulate error. This project is for academic demonstration and requires validation before real business use.

### Future Scope
Compare multiple forecasting models, add holiday and promotion variables, forecast by product or region, and retrain using newly collected data.

### Conclusion
The project demonstrates an end-to-end sales forecasting workflow with manual dataset upload, preprocessing, machine learning, evaluation, visualization, and future prediction.
