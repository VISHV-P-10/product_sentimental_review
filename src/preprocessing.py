
import re
import unicodedata


def clean_text(text):
    """
    Basic preprocessing for Hinglish social-media text.

    Preserves:
    - Emojis
    - Negations
    - Hindi transliteration
    - Sentiment-bearing words
    """

    if not isinstance(text, str):
        return ""

    # Normalize Unicode representation
    text = unicodedata.normalize("NFC", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove hashtag symbol but preserve the word
    text = text.replace("#", "")

    # Normalize elongated characters:
    # "sooooo" -> "sooo"
    text = re.sub(
        r"(.)\1{3,}",
        r"\1\1\1",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


if __name__ == "__main__":

    examples = [
        "YAAAAAR this movie is AMAZING 😍😍",
        "Mujhe ye movie bilkul pasand nahi!!!",
        "Check this https://example.com",
        "@rahul #Amazing movie hai",
        "This is sooo boringgggg"
    ]

    for example in examples:
        print("Original:", example)
        print("Cleaned :", clean_text(example))
        print("-" * 50)
