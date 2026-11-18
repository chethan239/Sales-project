# Sales Forecasting & Predictive Analytics Mini Project

A Python + Machine Learning mini project with a Streamlit dashboard and manual CSV upload.

## Features
- Upload your own CSV directly in the Streamlit sidebar.
- Use the included sample CSV if no file is uploaded.
- Validate required columns and clean invalid rows.
- Visualize monthly sales.
- Train a Random Forest Regression model.
- Evaluate with MAE, RMSE, and R² on the final six chronological observations.
- Forecast 1–12 future months.
- Download forecast results as CSV.

## Folder structure
```text
Sales_Forecasting_Mini_Project_Manual_CSV/
├── app.py
├── forecasting.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── VIVA_QUESTIONS.md
└── data/
    ├── sales_data.csv
    └── DATA_DICTIONARY.md
```

## Deploy to Streamlit Community Cloud
1. Extract this ZIP.
2. Upload the project files and folders to a GitHub repository.
3. Ensure `app.py`, `forecasting.py`, `requirements.txt`, and `data/sales_data.csv` are in the repository at the shown paths.
4. In Streamlit Community Cloud, deploy the repository and set the main file path to `app.py`.
5. Open the app. Use **Upload sales CSV** in the left sidebar to upload your own dataset.

## Run locally
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run:
```bash
streamlit run app.py
```

## Required CSV format
The CSV must contain these exact column names:

| Date | Sales |
|---|---:|
| 2024-01-01 | 45000 |
| 2024-02-01 | 52000 |
| 2024-03-01 | 48000 |

- `Date`: dates that pandas can parse.
- `Sales`: numeric, non-negative sales values.
- Provide at least 15 months of monthly data; more history is recommended.
- Additional columns are allowed but are not currently used by this model.
- If there are multiple transactions in a month, the app sums them into monthly sales.

## Important
The included 60-month dataset is synthetic sample data for demonstration. Replace it with real data for meaningful business conclusions. Forecasts are estimates, not guaranteed future sales.
