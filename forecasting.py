from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
MODEL_PATH = ROOT / "artifacts" / "sales_model.joblib"


def load_data(source=None):
    """Load and validate monthly sales data from a CSV path or file-like object."""
    source = DATA_PATH if source is None else source
    df = pd.read_csv(source)

    required = {"Date", "Sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV must contain columns named Date and Sales. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"])

    if df.empty:
        raise ValueError("No valid Date and Sales rows were found.")
    if (df["Sales"] < 0).any():
        raise ValueError("Sales values cannot be negative.")

    monthly = (
        df.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )
    return monthly.dropna()


def make_features(series):
    frame = pd.DataFrame({"Sales": series})
    frame["month"] = frame.index.month
    frame["year"] = frame.index.year
    frame["lag_1"] = frame["Sales"].shift(1)
    frame["lag_2"] = frame["Sales"].shift(2)
    frame["lag_3"] = frame["Sales"].shift(3)
    frame["rolling_mean_3"] = frame["Sales"].shift(1).rolling(3).mean()
    return frame.dropna()


def train_and_evaluate(source=None):
    monthly = load_data(source)
    series = monthly.set_index("Date")["Sales"].asfreq("MS")

    # Fill months with no recorded sales as zero to keep a continuous monthly series.
    series = series.fillna(0)
    features = make_features(series)

    if len(features) < 12:
        raise ValueError(
            "Please provide at least 15 months of monthly sales data "
            "so the model has enough training and test records."
        )

    X = features.drop(columns="Sales")
    y = features["Sales"]

    # Keep the final six observations as a chronological test set.
    split = max(1, len(features) - 6)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan")
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    comparison = pd.DataFrame({
        "Date": y_test.index,
        "Actual_Sales": y_test.values,
        "Predicted_Sales": predictions
    })
    return model, metrics, comparison, series


def forecast_next_months(model, history, periods=6):
    history = history.copy().asfreq("MS").fillna(0)
    results = []

    for _ in range(periods):
        next_date = history.index[-1] + pd.offsets.MonthBegin(1)
        row = pd.DataFrame([{
            "month": next_date.month,
            "year": next_date.year,
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "rolling_mean_3": history.iloc[-3:].mean()
        }], index=[next_date])

        value = max(0.0, float(model.predict(row)[0]))
        results.append({"Date": next_date, "Forecast_Sales": value})
        history.loc[next_date] = value

    return pd.DataFrame(results)
