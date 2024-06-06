
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

file_path = 'cleaned_reviews_textblob.csv'
reviews_df = pd.read_csv(file_path)

print(reviews_df.head())

sentiment_counts = reviews_df['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Intel Product Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()

print(sentiment_counts)

