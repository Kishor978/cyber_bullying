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

# Ordered emotion labels from the model
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def extract_features(text, max_length=128):
    """Extracts VADER sentiment and emotion vector (7D) from text safely."""
    initialize_emotion_analyzer()

    try:
        text = limit_emoji_repeats(text)
        text_with_emojis = emoji.demojize(text)

        # VADER sentiment
        vader_score = vader_analyzer.polarity_scores(text_with_emojis)['compound']
    except Exception as e:
        print(f"[❌ VADER failed] {text[:80]}... | {e}")
        vader_score = 0.0

    emotion_vector = [0.0] * len(EMOTION_LABELS)

    if emotion_pipeline:
        try:
            # Truncate long text using tokenizer from pipeline
            tokenizer = emotion_pipeline.tokenizer
            if tokenizer:
                encoded = tokenizer(text_with_emojis, truncation=True, max_length=max_length, return_tensors="pt")
                decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            else:
                decoded = text_with_emojis

            emotions = emotion_pipeline(decoded)
            # print(f"[DEBUG] text: {decoded[:80]}...\n[DEBUG] pipeline output: {emotions}\n")

            if not emotions or not isinstance(emotions[0], list):
                raise ValueError("Empty or invalid emotion output.")

            # Convert list of dicts to dict for safe access
            emotion_scores = {e['label']: e['score'] for e in emotions[0]}

            # Build vector in fixed label order
            emotion_vector = [emotion_scores.get(label, 0.0) for label in EMOTION_LABELS]

            # Sanity check
            if len(emotion_vector) != len(EMOTION_LABELS) or np.any(np.isnan(emotion_vector)) or np.any(np.isinf(emotion_vector)):
                raise ValueError("Invalid emotion vector content.")

        except Exception as e:
            print(f"[⚠️ Emotion extraction failed] {text[:80]}... | Reason: {e}")
            emotion_vector = [0.0] * len(EMOTION_LABELS)

    return vader_score, emotion_vector

def process_texts_for_emotion_features(df, text_column='text'):
    """Applies emotion and sentiment extraction to each row safely."""
    emoji_scores = []
    emotion_vectors = []

    print("🔍 Extracting emotion and sentiment features...")
    for t in tqdm(df[text_column], desc="Extracting Emotion Features"):
        score, vector = extract_features(t)
        emoji_scores.append(score)
        emotion_vectors.append(vector)

    df['emoji_score'] = emoji_scores
    df['emotion_vector'] = emotion_vectors

    # --- Determine expected emotion vector length dynamically ---
    valid_lengths = df['emotion_vector'].apply(lambda x: isinstance(x, list)).sum()
    vector_lengths = df['emotion_vector'].apply(lambda x: len(x) if isinstance(x, list) else -1)
    most_common_length = vector_lengths[vector_lengths != -1].mode().iloc[0] if not vector_lengths.empty else 0

    print(f"✅ Detected most common emotion vector length: {most_common_length}")

    # --- Clean data ---
    # Keep only rows with valid emoji score
    df = df[df['emoji_score'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]

    # Keep only rows with valid-length, finite emotion vectors
    df = df[df['emotion_vector'].apply(
        lambda x: isinstance(x, list)
        and len(x) == most_common_length
        and all(np.isfinite(xi) for xi in x)
    )]

    print(f"🧹 After filtering: {df.shape[0]} valid rows retained.")
    return df

import re

def limit_emoji_repeats(text, max_repeat=5):
    emoji_pattern = r'(([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF])+)\2{' + str(max_repeat) + ',}'
    return re.sub(emoji_pattern, r'\1'*max_repeat, text)



test_texts = [
    "Hell is hot 🔥🔥🔥🔥 and all you fucking woman are not!",
    "just let the refugees in 😤😤😤",
    "just DON'T let the refugees in then 😤😤😤"
]

for text in test_texts:
    s, v = extract_features(text)
    print(f"Text: {text}\nVADER: {s}\nEmotion Vector: {v}\n{'-'*60}")
