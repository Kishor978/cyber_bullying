# src/text_preprocessing.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy # For BiLSTM notebook's tokenization

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Performs basic text cleaning: lowercasing, link/mention/hashtag removal, punctuation/number removal."""
    text = text.lower() #
    text = re.sub(r"http\S+|www\S+|https\S+", '', text) # remove links
    text = re.sub(r'@\w+|#\w+', '', text)                # remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)                 # remove numbers, punctuations
    return text

def clean_and_tokenize(text): # From bilstm.ipynb
    """Cleans text and tokenizes using spaCy, removing stop words and punctuation."""
    text = text.lower() #
    text = re.sub(r"http\S+|www\S+", '', text) #
    text = re.sub(r"[^\w\s]", '', text) #
    doc = nlp(text) #
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct] #
    return tokens

def vectorize_tfidf(texts, max_features):
    """Applies TF-IDF vectorization to a list of texts."""
    tfidf = TfidfVectorizer(max_features=max_features) #
    X = tfidf.fit_transform(texts).toarray() #
    return X, tfidf # Return tfidf object for potential later use (e.g., transforming new text)