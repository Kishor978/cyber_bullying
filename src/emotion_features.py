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

def extract_features(text): #
    """Extracts VADER sentiment and emotion vectors from text."""
    initialize_emotion_analyzer() # Ensure analyzer is loaded

    text_with_emojis = emoji.demojize(text) #

    vader_score = vader_analyzer.polarity_scores(text_with_emojis)['compound'] #

    emotion_vector = []
    if emotion_pipeline: #
        try: #
            emotions = emotion_pipeline(text_with_emojis) #
            emotion_vector = [e['score'] for e in emotions[0]] #
        except Exception as e: #
            print(f"Emotion analysis for text '{text}' failed: {e}") #
            emotion_vector = [0.0] * 6  # Assuming 6 emotion classes as placeholder
    else:
        emotion_vector = [0.0] * 6 # Placeholder if pipeline failed to load

    return vader_score, emotion_vector

def process_texts_for_emotion_features(df, text_column='text'):
    """Applies feature extraction to a DataFrame column."""
    emoji_scores = []
    emotion_vectors = []

    for t in tqdm(df[text_column], desc="Extracting Emotion Features"): #
        s, e = extract_features(t) #
        emoji_scores.append(s) #
        emotion_vectors.append(e) #

    df['emoji_score'] = emoji_scores #
    df['emotion_vector'] = emotion_vectors #
    return df