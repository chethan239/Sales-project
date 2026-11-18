from forecasting import train_and_evaluate

if __name__ == "__main__":
    model, metrics, comparison, _ = train_and_evaluate()
    comparison.to_csv("artifacts/test_predictions.csv", index=False)
    print("Model trained. Model file: artifacts/sales_model.joblib")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:,.3f}")
    print(comparison.to_string(index=False))
