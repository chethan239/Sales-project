import io
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecasting import load_data, train_and_evaluate, forecast_next_months

st.set_page_config(
    page_title="Sales Forecasting & Predictive Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Predictive Analytics")
st.write(
    "Upload your own monthly sales CSV or use the included sample dataset. "
    "The model uses Random Forest Regression."
)

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload sales CSV",
        type=["csv"],
        help="Required columns: Date and Sales. Use at least 15 months of data."
    )
    horizon = st.slider("Forecast horizon (months)", 1, 12, 6)

# Use fresh in-memory streams for each read. Streamlit UploadedFile has a
# mutable cursor, so reusing the same stream without rewinding can cause EOF errors.
if uploaded_file is not None:
    source_bytes = uploaded_file.getvalue()
    source_for_preview = io.BytesIO(source_bytes)
    source_for_model = io.BytesIO(source_bytes)
    dataset_label = uploaded_file.name
else:
    source_for_preview = None
    source_for_model = None
    dataset_label = "Included sample dataset"

try:
    monthly = load_data(source_for_preview)

    if len(monthly) < 15:
        st.error(
            f"Only {len(monthly)} monthly records were found. "
            "Please upload at least 15 months of data."
        )
        st.stop()

    model, metrics, comparison, series = train_and_evaluate(source_for_model)
    forecast = forecast_next_months(model, series, horizon)

except Exception as exc:
    st.error(f"Could not run the model: {exc}")
    st.info("Your CSV must contain valid Date and Sales columns, with at least 15 months of data.")
    st.stop()

st.success(f"Dataset loaded: {dataset_label}")

col1, col2, col3 = st.columns(3)
col1.metric("Total historical sales", f"₹{monthly['Sales'].sum():,.0f}")
col2.metric("Average monthly sales", f"₹{monthly['Sales'].mean():,.0f}")
col3.metric("Months of data", f"{len(monthly)}")

st.subheader("Historical monthly sales")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly["Date"], monthly["Sales"], marker="o", linewidth=1.8)
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Model performance — final six test months")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"₹{metrics['MAE']:,.2f}")
m2.metric("RMSE", f"₹{metrics['RMSE']:,.2f}")
m3.metric("R²", f"{metrics['R2']:.3f}" if pd.notna(metrics["R2"]) else "N/A")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(comparison["Date"], comparison["Actual_Sales"], marker="o", label="Actual")
ax.plot(comparison["Date"], comparison["Predicted_Sales"], marker="o", label="Predicted")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader(f"Forecast for the next {horizon} months")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(series.index, series.values, label="Historical")
ax.plot(forecast["Date"], forecast["Forecast_Sales"], marker="o", linestyle="--", label="Forecast")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(
    forecast.style.format({"Forecast_Sales": "₹{:,.2f}"}),
    use_container_width=True
)
st.download_button(
    "Download forecast CSV",
    data=forecast.to_csv(index=False).encode("utf-8"),
    file_name="sales_forecast.csv",
    mime="text/csv"
)

with st.expander("Preview uploaded / sample dataset"):
    st.dataframe(monthly, use_container_width=True)

st.caption(
    "Note: The included dataset is synthetic demonstration data. "
    "Forecast quality depends on the quality and representativeness of your own data."
)
