import pandas as pd

train_path = r"E:\product_setimental_project\data\processed\train.csv"
dev_path = r"E:\product_setimental_project\data\processed\dev.csv"

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)

print("TRAIN SHAPE:", train_df.shape)
print("DEV SHAPE:", dev_df.shape)

print("\nFIRST FIVE EXAMPLES")
print(train_df.head())

print("\nDATASET INFORMATION")
print(train_df.info())

print("\nMISSING VALUES")
print(train_df.isnull().sum())

print("\nDUPLICATES")
print(train_df.duplicated(subset=["text"]).sum())

print("\nSENTIMENT DISTRIBUTION")
print(train_df["label"].value_counts())

train_df["word_count"] = (
    train_df["text"].astype(str).apply(
        lambda x: len(x.split())
    )
)

print("\nAVERAGE SENTENCE LENGTH:")
print(train_df["word_count"].mean())