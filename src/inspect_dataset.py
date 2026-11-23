from pathlib import Path
DATA_DIR = Path("data/raw")
files = {
    "Hinglish_train_14k_split_conll.txt",
    "Hinglish_dev_3k_split_conll.txt"
}
# looping over the files
for filename in files:
    file_path = DATA_DIR/filename

    # printign a visual seperator
    print("\n"+ "=" * 60)
    print("FILE:,filename")
    print("="*60)

    #Safety check
    if not file_path.exists():
        print("File not found:", file_path)
        continue

    # Reading the File
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for i, line in enumerate(file):

            if i == 35:
                break #Stops after 36 lines 

            print(repr(line.rstrip("\n"))) #Removes trailing newline
