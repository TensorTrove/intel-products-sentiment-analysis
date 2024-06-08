import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from textblob import TextBlob

# Load the CSV file
file_path = './reviews.csv'
reviews_df = pd.read_csv(file_path)

# Initialize the Helsinki-NLP/opus-mt-mul-en translation model
model_name = 'Helsinki-NLP/opus-mt-mul-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate reviews to English
def translate_review(review):
    try:
        # Tokenize the text
        tokens = tokenizer(review, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        # Translate
        translated_tokens = model.generate(**tokens)
        # Decode the translated text
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return review

# Translate non-English reviews
reviews_df['translated_content'] = reviews_df['content'].fillna('').apply(translate_review)

# Function to determine sentiment
def get_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
reviews_df['sentiment'] = reviews_df['translated_content'].apply(get_sentiment)

# Save the new DataFrame to a CSV file
output_file_path = 'translated_reviews_with_sentiment.csv'
reviews_df[['translated_content', 'sentiment']].to_csv(output_file_path, index=False)
