# src/emotion_features.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import emoji
from tqdm.auto import tqdm # Use tqdm.auto for notebook/script compatibility

vader_analyzer = SentimentIntensityAnalyzer()
emotion_pipeline = None

def initialize_emotion_analyzer():
    global emotion_pipeline
    if emotion_pipeline is None:
        model_name = "j-hartmann/emotion-english-distilroberta-base" #
        try:
            # You might need to ensure model files are cached or downloaded before this step
            # cached_file("j-hartmann/emotion-english-distilroberta-base", "config.json", force_download=True)
            emotion_pipeline = pipeline( #
                "text-classification", #
                model=model_name, #
                return_all_scores=True, #
                framework="pt" #
            )
            print("✅ Emotion analysis pipeline loaded successfully!")
        except Exception as e: #
            print(f"❌ Failed to load emotion model: {e}") #
            emotion_pipeline = None

import numpy as np

def extract_features(text, max_length=128):
    """Extracts VADER sentiment and emotion vectors from text safely."""
    initialize_emotion_analyzer()  # Ensure pipeline is loaded

    try:
        # Limit emoji spam
        text = limit_emoji_repeats(text)

        # Demojize for VADER input
        text_with_emojis = emoji.demojize(text)
        vader_score = vader_analyzer.polarity_scores(text_with_emojis)['compound']
    except Exception as e:
        print(f"[❌ VADER failed] Text: {text[:80]}... | Reason: {e}")
        vader_score = 0.0

    emotion_vector = [0.0] * 6  # Default vector with 6 zeros

    if emotion_pipeline:
        try:
            # Truncate using tokenizer if available
            tokenizer = emotion_pipeline.tokenizer
            if tokenizer:
                encoded = tokenizer(text_with_emojis, truncation=True, max_length=max_length, return_tensors="pt")
                decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
                emotions = emotion_pipeline(decoded)
            else:
                emotions = emotion_pipeline(text_with_emojis)

            emotion_vector = [e['score'] for e in emotions[0]]

            # Sanity check
            if len(emotion_vector) != 6 or np.any(np.isnan(emotion_vector)) or np.any(np.isinf(emotion_vector)):
                raise ValueError("Invalid emotion vector output.")

        except Exception as e:
            print(f"[⚠️ Emotion extraction failed] Text: {text[:80]}... | Reason: {e}")
            emotion_vector = [0.0] * 6

    return vader_score, emotion_vector

def process_texts_for_emotion_features(df, text_column='text'):
    """Applies emotion and sentiment extraction to each row safely."""
    emoji_scores = []
    emotion_vectors = []

    for t in tqdm(df[text_column], desc="Extracting Emotion Features"):
        score, vector = extract_features(t)
        emoji_scores.append(score)
        emotion_vectors.append(vector)

    df['emoji_score'] = emoji_scores
    df['emotion_vector'] = emotion_vectors

    # Remove bad rows with invalid vectors or emoji score
    df = df[df['emoji_score'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
    df = df[df['emotion_vector'].apply(lambda x: isinstance(x, list) and len(x) == 6 and not any(np.isnan(xi) or np.isinf(xi) for xi in x))]

    return df

import re

def limit_emoji_repeats(text, max_repeat=5):
    emoji_pattern = r'(([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF])+)\2{' + str(max_repeat) + ',}'
    return re.sub(emoji_pattern, r'\1'*max_repeat, text)
