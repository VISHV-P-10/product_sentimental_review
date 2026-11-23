from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

#error handling, im creatin prosessed folder if there's not presented
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def convert_conll_to_dataframe(file_path):
    #This function reads one input file and returns a pandas DataFrame with two columns: text and label.

    records = []
    tokens = []
    sentiment = None

    def save_tweet():
        nonlocal tokens,sentiment
        text = " ".join(tokens).strip()

        if text and sentiment in {
            "positive",
            "negative",
            "neutral"
        }:
            records.append({
                "text": text,
                "label": sentiment
            })

        tokens =[]
        sentiment = None

    with open(
        file_path,"r",encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            #blank line : end of tweet
            if not line:

                save_tweet()
                continue
            columns = line.split()
            if columns[0].lower()=="meta":
                save_tweet()
                candidate = columns[-1].lower()

                if candidate in {
                    "positive",
                    "negative",
                    "neutral"
                }:

                    sentiment = candidate

            else:

                # First column is the actual word/token
                if sentiment is not None:

                    tokens.append(columns[0])

    save_tweet()
    return pd.DataFrame(records)

def main():
    datasets = {
        "train": "Hinglish_train_14k_split_conll.txt",
        "dev": "Hinglish_dev_3k_split_conll.txt"
    }

    for split, filename in datasets.items():

        input_path = DATA_DIR / filename

        if not input_path.exists():

            raise FileNotFoundError(input_path)

        df = convert_conll_to_dataframe(input_path)

        output_path = OUTPUT_DIR / f"{split}.csv"

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

        print(f"\n{split.upper()} DATASET")
        print("Shape:", df.shape)
        print("\nFirst five rows:")
        print(df.head())

        print("\nLabel distribution:")
        print(df["label"].value_counts())

        print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
