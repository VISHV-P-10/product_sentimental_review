
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "tfidf"

dev_df = pd.read_csv(DATA_DIR / "dev_clean.csv")

model = joblib.load(
    MODEL_DIR / "logistic_regression.joblib"
)

X_dev = dev_df["cleaned_text"].fillna("").astype(str)
y_dev = dev_df["label"]

predictions = model.predict(X_dev)

results = dev_df[["text", "label"]].copy()

results["predicted_label"] = predictions

# Extract incorrect predictions
errors = results[
    results["label"] != results["predicted_label"]
].copy()

print("Total incorrect predictions:", len(errors))

print("\nError distribution:")
print(
    errors.groupby(
        ["label", "predicted_label"]
    ).size().sort_values(ascending=False)
)

print("\nSample incorrect predictions:")

for _, row in errors.head(20).iterrows():

    print("\nText:", row["text"])
    print("Actual:", row["label"])
    print("Predicted:", row["predicted_label"])
    print("-" * 60)

# Save errors for further investigation
errors.to_csv(
    ROOT / "results" / "misclassified_tweets.csv",
    index=False
)

print("\nSaved misclassified tweets.")
