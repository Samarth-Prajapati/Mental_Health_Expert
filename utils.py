import json
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# For translation and transliteration
from googletrans import Translator
from transliterate import translit

# Utility to translate English to Hindi (in English script)
def to_hinglish(text):
    translator = Translator()
    # Translate to Hindi
    hindi = translator.translate(text, dest='hi').text
    # Transliterate Hindi to Latin (English script)
    try:
        # Some environments may not support transliterate for Hindi, fallback to original
        hinglish = translit(hindi, 'hi', reversed=True)
    except Exception:
        hinglish = hindi
    return hinglish

# Load quotes
def load_quotes():
    with open('quotes.json', 'r') as file:
        return json.load(file)

# Sentiment analysis model
class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()
        # Sample training data (in real-world, use a larger dataset)
        self.train_model()

    def train_model(self):
        # Sample data for training
        texts = [
            "I feel amazing today!", "This is the best day ever!",
            "I'm so sad and lonely.", "Everything feels hopeless.",
            "Just another normal day.", "Nothing special happening."
        ]
        labels = ['positive', 'positive', 'negative', 'negative', 'neutral', 'neutral']
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        X = self.vectorizer.transform([text])
        predicted_label = self.model.predict(X)[0]
        
        # Combine TextBlob and ML model for better accuracy
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return predicted_label, polarity

# Mood tracking
class MoodTracker:
    def __init__(self):
        self.mood_history = []

    def update_mood(self, sentiment, polarity):
        self.mood_history.append({'sentiment': sentiment, 'polarity': polarity})

    def summarize_mood(self):
        if not self.mood_history:
            return "neutral", "No conversation yet to analyze."
        
        df = pd.DataFrame(self.mood_history)
        avg_polarity = df['polarity'].mean()
        sentiment_counts = df['sentiment'].value_counts()
        dominant_mood = sentiment_counts.idxmax()
        
        if avg_polarity > 0.1:
            behavior = "You seem optimistic and engaged."
        elif avg_polarity < -0.1:
            behavior = "You might be feeling a bit down or stressed."
        else:
            behavior = "Your mood appears balanced and neutral."
        
        return dominant_mood, behavior

# Get motivational quote based on mood
def get_quote(mood):
    quotes = load_quotes()
    return np.random.choice(quotes.get(mood, quotes['neutral']))