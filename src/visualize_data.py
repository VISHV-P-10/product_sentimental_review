
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/train.csv")

sentiment_counts = df["label"].value_counts()

plt.figure(figsize=(8, 5))

sentiment_counts.plot(
    kind="bar"
)

plt.title("Hinglish Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "results/sentiment_distribution.png"
)

plt.show()
