
from pathlib import Path
import pandas as pd

from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"

def process_dataset(input_file, output_file):

    df = pd.read_csv(input_file)

    # Validate required columns
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            f"Expected text and label columns in {input_file}"
        )

    # Drop rows with missing text or labels
    df = df.dropna(subset=["text", "label"]).copy()

    # Preserve original text and create cleaned text
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Remove empty text rows
    df = df[df["cleaned_text"].str.strip() != ""]

    # Save
    df.to_csv(output_file, index=False)

    print(f"\nProcessed: {input_file.name}")
    print("Shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("Saved:", output_file)


if __name__ == "__main__":

    process_dataset(
        DATA_DIR / "train.csv",
        DATA_DIR / "train_clean.csv"
    )

    process_dataset(
        DATA_DIR / "dev.csv",
        DATA_DIR / "dev_clean.csv"
    )
