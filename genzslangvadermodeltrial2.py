import pandas as pd
from datasets import load_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Loading the Dataset from Hugging Face
print("Loading dataset...")
dataset = load_dataset("MLBtrio/genz-slang-dataset", split="train")
df = dataset.to_pandas()

# Preprocessing the Dataset
print("Preprocessing dataset...")
# Standardizing column names
df = df.rename(columns={"Slang": "word", "Description": "meaning", "Example": "example"})

# Removing duplicates
df = df.drop_duplicates(subset=['word'])

# lowercase for slang words
df['word'] = df['word'].str.lower()

# custom lexicon for some brainrot words and their sentiment scores
custom_lexicon = {
    "w": 0.9,
    "l": -0.8,
    "dank": -0.3,
    "cheugy": -0.3,
    "bop": 0.9,
    "goat": 0.9,
    "salty": -0.8,
    "drip": 0.0,
    "bussin'": 0.9,
    "snatched": 0.8,
    "ffs": -0.5,
    "cap": -0.7,
    "no cap": 0.7,
    "snack": 0.6,
    "sheesh": 0.7,
    "bet": 0.2,
    "based": 0.5,
    "cringe": -0.8,
    "mid": -0.3,
    "npc": -0.5,
    "ratio": -0.8,
    "slaps": 0.8,
    "sus": -0.6,
    "slay": 0.9,
    "vibing": 0.8,
    "chad": 0.8,
    "clapped": -0.7,
    "clown": -0.9,
    "karen": -0.7,
    "mommy": 0.7,
    "pick me": -0.6,
    "shade": -0.5,
    "smh": -0.2,
    "yas": 0.9,
    "ick": -0.8,
    "ate": 1.0,
    "heather": 0.9,
    "skill-issue": -0.7,
    "benching": -0.2,
    "gaslighting": -0.7,
    "bae": 0.8,
    "gal": -0.8,
    "badass": 0.8,
    "basic": -0.5,
    "dilligaf": -0.1,
    "jk": 0.4,
    "noob": -0.6,
    "rizz": 0.0,
    "gyatt": 0.6,
    "fam": 0.8,
    "cancel": -0.5,
    "stan": 0.6,
    "e-boy": -0.2,
    "e-girl": -0.2,
    "dank": 0.5,
    "ghost": -0.1,
    "big yikes": -0.3,
    "simp": 0.0,
    "snack": 0.6,
    "drip": 0.8,
    "banger": 0.7,
    "bet": 0.3,
    "sending": 0.6,
    "alpha": 1.0,
    "skibidi toilet": -0.9,
    "ohio": -0.8,
    "alabama": -0.7,
    "edging": -0.6,
    "fanum tax": -0.3,
    "mewing": 0.9,
    "mog": 0.9,
    "mogger": 0.9,

}

# Converting the processed data into a dictionary for VADER
df['sentiment_score'] = df['word'].apply(lambda x: custom_lexicon.get(x, 0.0))  # Default to 0.0 if word not found

# Extending VADER’s Lexicon
print("Extending VADER's lexicon...")
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(custom_lexicon)


# Defining a function to assign sentiment based on context
def assign_sentiment_based_on_context(row):
    """Assign sentiment to the example based on context using VADER."""
    # Analyze sentiment of the example context
    context_sentiment = analyzer.polarity_scores(row['example'])['compound']

    # Use the custom lexicon for word-based sentiment (if available), else fallback to context sentiment
    word_sentiment = custom_lexicon.get(row['word'], context_sentiment)

    # Combine word sentiment and context sentiment
    final_sentiment = (0.7 * word_sentiment) + (0.3 * context_sentiment)  # Adjust weights if needed
    return final_sentiment


# Applying context-based sentiment analysis for each row in the dataset
print("Assigning sentiment based on context...")
df['final_sentiment'] = df.apply(assign_sentiment_based_on_context, axis=1)

# Evaluating with Dataset Examples
print("\nEvaluating with dataset examples...")
print(df[['word', 'example', 'final_sentiment']].head())

# Visualizing Sentiment Distribution
print("\nVisualizing sentiment distribution...")
plt.hist(df['final_sentiment'], bins=20, color='blue', edgecolor='black')
plt.title("Sentiment Distribution of Examples")
plt.xlabel("Compound Sentiment Score")
plt.ylabel("Frequency")
plt.show()


# Real-Time Sentiment Analysis
def analyze_text(text):
    """Analyze sentiment for the given text."""
    return analyzer.polarity_scores(text)


print("\n whats good y'all! This is my first ever sentiment analysis model. (Ik,big yikes but check it out anyway).")
while True:
    user_input = input("Enter a sentence, preferably in genZ (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("aight thanks bro, peace out!")
        break
    sentiment_result = analyze_text(user_input)
    print(f"Input: {user_input}")
    print(f"Sentiment: {sentiment_result}")
