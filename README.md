![Logo](https://github.com/Terminal127/sen-analysis-intel/blob/main/Default_sentimental_analysis_using_machine_learning_0.jpg)


# Intel Products Sentiment Analysis

This project analyzes customer sentiment towards Intel products using a multi-stage approach.

## Project Overview

1. **Data Gathering**: Web scraping customer reviews from Amazon using Selenium.

2. **Sentiment Analysis & Visualization**:
   - Data cleaning
   - Dual sentiment analysis: NLTK's VADER and Hugging Face Transformers (BERTweet)
   - Visualization of sentiment distribution

3. **Aspect-Based Sentiment Analysis (ABSA)**:
   - Identify problematic aspects in negative reviews using spaCy
   - Analyze aspect-specific sentiment with a fine-tuned DeBERTa model

4. **Insights and Actionable Recommendations**:
   - Provide deep understanding of customer feedback
   - Guide product improvements, marketing strategies, and customer support

## Documentation

### Python Script:

The script defines a main function that takes various parameters such as the URL of the Amazon product review page, the number of pages to scrape, the output CSV file name, wait time for elements to load, the number of retries, and an optional proxy server address.

- The script sets up a headless browser using Selenium to navigate to the given URL and scrape reviews from the specified number of pages.
- The scraped reviews are saved to a CSV file.

### links.txt File:

This file contains a list of URLs, each pointing to an Amazon product review page.


## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/TensorTrove/intel-products-sentiment-analysis.git
   ```

2. Navigate to the project directory:
   ```sh
   cd intel-products-sentiment-analysis
   ```

3. Run the python file:
   ```sh
   python3 comments.py
   # it will generate reviews.csv
   ```

4. Clean the data:
   ```sh
   python3 clean_reviews.py
   # it will generate cleaned_reviews_textblob.csv
   ```

5. Run the Jupyter Notebooks for different models:
   - Intel_Products_Sentiment_Analysis_CNN.ipynb
   - Intel_Products_Sentiment_Analysis_FastText.ipynb
   - Intel_Products_Sentiment_Analysis_LSTM.ipynb
   - Intel_Products_Sentiment_Analysis_NLTK.ipynb
   - Intel_Products_Sentiment_Analysis_NLTK_With_translation.ipynb
   - Intel_Products_Sentiment_Analysis_RoBERTa.ipynb
   - Intel_Products_Sentiment_Analysis_ULMFiT.ipynb
   - Intel_Products_Sentiment_Analysis_VADER.ipynb

6. Run the ASBA for Sentiment Analysis: Analyze the sentiment of reviews using the DeBERTa model, focusing on specific aspects of Intel products:
- Performance
- Battery
- Design
- Value

**Example:**
   ```sh
   jupyter notebook Intel_Products_Aspect_Based_Sentiment_Analysis_DeBERTa.ipynb
   ```
   

## Contributing

Contributions are always welcome!

## Recent Updates

- Added CNN model for sentiment analysis
- Implemented FastText and LSTM models
- Added NLTK analysis with and without translation
- Incorporated RoBERTa model for sentiment analysis
- Implemented ULMFiT model
- Added VADER sentiment analysis

## Authors

- [@ankitdey-marsh](https://www.github.com/ankitdey-marsh)
- [@Terminal127](https://www.github.com/Terminal127)
- [@debjit-mandal](https://www.github.com/debjit-mandal)
- [@nilotpal-basu](https://www.github.com/nilotpal-basu)
- [@MrCelestial](https://www.github.com/MrCelestial)

------------------------------------------------------------------------------------------------------------

Feel free to explore and run these notebooks to compare different sentiment analysis approaches.


