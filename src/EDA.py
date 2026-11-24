
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

#RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
dev_df = pd.read_csv(DATA_DIR / "dev.csv")

# Combine for exploratory analysis
df = pd.concat([train_df, dev_df], ignore_index=True)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nData information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate tweets:", df["text"].duplicated().sum())

print("\nSentiment distribution:")
print(df["label"].value_counts())

# Remove missing text only for analysis
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)

# Tweet length
df["char_length"] = df["text"].str.len()

df["word_count"] = df["text"].apply(
    lambda text: len(text.split())
)

print("\nTweet length statistics:")
print(df[["char_length", "word_count"]].describe())

print("\nAverage words per sentiment:")
print(df.groupby("label")["word_count"].mean())

# 1. Tweet length distribution
plt.figure(figsize=(9, 5))

plt.hist(
    df["word_count"],
    bins=30,
    edgecolor="black"
)

plt.title("Hinglish Tweet Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Tweets")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "tweet_length_distribution.png"
)
plt.close()

# 2. Average tweet length by sentiment
avg_length = df.groupby("label")["word_count"].mean()

plt.figure(figsize=(8, 5))
avg_length.plot(kind="bar", edgecolor="black")

plt.title("Average Tweet Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Number of Words")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "average_tweet_length.png"
)
plt.close()

# 3. Most frequent words
def get_top_words(texts, n=20):
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_features=5000
    )

    matrix = vectorizer.fit_transform(texts)

    frequencies = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    counts = Counter(dict(zip(words, frequencies)))

    return counts.most_common(n)


top_words = get_top_words(df["text"].tolist())

print("\nTop 20 frequent words:")
for word, count in top_words:
    print(f"{word}: {count}")

words, counts = zip(*top_words)

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], counts[::-1])

plt.title("Top 20 Frequent Words")
plt.xlabel("Frequency")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "top_words.png")
plt.close()

# 4. Most frequent bigrams
vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    max_features=5000,
    token_pattern=r"(?u)\b\w+\b"
)

bigram_matrix = vectorizer.fit_transform(df["text"])

bigram_counts = bigram_matrix.sum(axis=0).A1
bigrams = vectorizer.get_feature_names_out()

top_bigrams = sorted(
    zip(bigrams, bigram_counts),
    key=lambda x: x[1],
    reverse=True
)[:20]

print("\nTop 20 bigrams:")
for bigram, count in top_bigrams:
    print(f"{bigram}: {count}")

print("\nEDA completed successfully!")
