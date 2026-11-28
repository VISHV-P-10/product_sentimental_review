
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "tfidf"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
dev_df = pd.read_csv(DATA_DIR / "dev_clean.csv")

TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "label"

X_train = train_df[TEXT_COLUMN].fillna("").astype(str)
y_train = train_df[LABEL_COLUMN].astype(str)

X_dev = dev_df[TEXT_COLUMN].fillna("").astype(str)
y_dev = dev_df[LABEL_COLUMN].astype(str)

# duplicate leakage check
train_texts = set(train_df["text"].dropna())
dev_texts = set(dev_df["text"].dropna())

overlap = train_texts.intersection(dev_texts)

print("Exact tweets appearing in both datasets:", len(overlap))

# Check that the datasets contain the expected classes
expected_labels = {"positive", "negative", "neutral"}

if set(y_train.unique()) != expected_labels:
    raise ValueError(
        f"Unexpected training labels: {set(y_train.unique())}"
    )

if set(y_dev.unique()) != expected_labels:
    raise ValueError(
        f"Unexpected development labels: {set(y_dev.unique())}"
    )

print("Training shape:", X_train.shape)
print("Development shape:", X_dev.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nDevelopment label distribution:")
print(y_dev.value_counts())


# --------------------------------------------------
# 3. Define TF-IDF + classifier pipelines
# --------------------------------------------------

def create_pipeline(classifier):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=30000,
                min_df=2,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])


models = {
    "Logistic Regression": create_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Linear SVM": create_pipeline(
        LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    )
}


# --------------------------------------------------
# 4. Train and evaluate
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    # Fit only on training data
    model.fit(X_train, y_train)

    # Predict on development data
    predictions = model.predict(X_dev)

    # Calculate metrics
    accuracy = accuracy_score(y_dev, predictions)

    precision = precision_score(
        y_dev,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_dev,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_dev,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_dev,
            predictions,
            labels=["negative", "neutral", "positive"],
            zero_division=0
        )
    )

    # Save model
    safe_name = model_name.lower().replace(" ", "_")

    model_path = MODEL_DIR / f"{safe_name}.joblib"

    joblib.dump(model, model_path)

    print("Saved model:", model_path)

    # Confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_dev,
        predictions,
        labels=["negative", "neutral", "positive"],
        display_labels=["Negative", "Neutral", "Positive"],
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{safe_name}_confusion_matrix.png"
    )

    plt.close()

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1
    })


# --------------------------------------------------
# 5. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="macro_f1",
        ascending=False
    ).to_string(index=False)
)

results_df.to_csv(
    RESULTS_DIR / "baseline_model_comparison.csv",
    index=False
)

with open(
    RESULTS_DIR / "baseline_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        results,
        file,
        indent=4
    )

print("\nBaseline training completed successfully!")
