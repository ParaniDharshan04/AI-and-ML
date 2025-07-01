import pandas as pd
from twitter_client import fetch_tweets
from preprocessing import clean_tweet
from sentiment import analyze_sentiment
from visualization import plot_pie, plot_bar

import os
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def main():
    keyword = input("Enter a keyword or hashtag to search for: ")
    max_tweets = int(input("Enter the number of tweets to fetch: "))
    print(f"Fetching tweets for '{keyword}'...")
    tweets = fetch_tweets(keyword, max_tweets)
    print(f"Fetched {len(tweets)} tweets.")

    print("Cleaning tweets...")
    cleaned = [clean_tweet(t) for t in tweets]

    print("Analyzing sentiment...")
    sentiments = [analyze_sentiment(t) for t in cleaned]

    df = pd.DataFrame({
        'Tweet': tweets,
        'Cleaned': cleaned,
        'Sentiment': sentiments
    })

    sentiment_counts = df['Sentiment'].value_counts().to_dict()
    print("Sentiment counts:", sentiment_counts)

    plot_pie(sentiment_counts)
    plot_bar(sentiment_counts)

    save = input("Save results to CSV? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"tweets_{keyword.replace('#','')}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    main() 