# Twitter Sentiment Analysis

A Python project to fetch tweets by keyword/hashtag, preprocess them, analyze sentiment, and visualize results.

## Features
- Fetch tweets using Tweepy
- Clean and preprocess tweet text
- Analyze sentiment (Positive, Neutral, Negative) using TextBlob
- Visualize results with pie and bar charts
- Optionally save results to CSV

## Setup
1. Clone the repo and navigate to the folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Fill in your Twitter API credentials in `config.py`.

## Usage
Run the main script:
```
python main.py
```
Follow the prompts to enter a keyword/hashtag and number of tweets.

## Notes
- Requires Twitter API v2 credentials (Bearer Token).
- For best results, use relevant keywords/hashtags.
- Results can be saved to CSV for further analysis. 