import pandas as pd
import re
from transformers import pipeline

file_path = 'reviews.csv'
reviews_df = pd.read_csv(file_path)

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\n', ' ', text)  
    text = re.sub(r'[^a-z0-9\s]', '', text)  
    return text

reviews_df['cleaned_content'] = reviews_df['content'].apply(clean_text)

sentiment_model = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

def analyze_sentiment_roberta(text):
    result = sentiment_model(text, truncation=True, max_length=512)
    return result[0]['label'].lower()

reviews_df['sentiment'] = reviews_df['cleaned_content'].apply(analyze_sentiment_roberta)

cleaned_reviews_df = reviews_df[['cleaned_content', 'sentiment']]

cleaned_csv_path = 'cleaned_reviews.csv'
cleaned_reviews_df.to_csv(cleaned_csv_path, index=False)

print(f"Cleaned reviews saved to {cleaned_csv_path}")
