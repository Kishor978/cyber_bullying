# src/text_preprocessing.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy # For BiLSTM notebook's tokenization
import re
import emoji

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans input text by:
    - Lowercasing
    - Removing links, mentions, hashtags, emojis
    - Removing non-letter characters
    - Removing extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = emoji.replace_emoji(text, replace='')           # remove emojis
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_and_tokenize(text):
    """
    Cleans text and tokenizes using spaCy, removing:
    - URLs
    - Punctuation
    - Stopwords
    - Emojis
    - Converts to lowercase
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)             # remove URLs
    text = emoji.replace_emoji(text, replace='')           # remove emojis
    text = re.sub(r"[^\w\s]", '', text)                    # remove punctuation
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return tokens


def vectorize_tfidf(texts, max_features):
    """Applies TF-IDF vectorization to a list of texts."""
    tfidf = TfidfVectorizer(max_features=max_features) #
    X = tfidf.fit_transform(texts).toarray() #
    return X, tfidf # Return tfidf object for potential later use (e.g., transforming new text)