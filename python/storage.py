from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reviews']
collection = db['statements']

df = pd.read_csv('translated_reviews_with_sentiment.csv')
def main()->None:
    for i in range(0,len(df)): 
        try:
            row = df.iloc[i]
            collection.insert_one({"Review":row['translated_content'],"Sentiment":row['sentiment']})
        except Exception as e:
            pass

if __name__=="__main__":
    main()
    print("Successfully Completed.")

