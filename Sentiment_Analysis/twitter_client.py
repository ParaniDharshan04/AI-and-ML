import tweepy
import time
from config import BEARER_TOKEN

def fetch_tweets(keyword, max_tweets=100):
    """
    Fetch recent tweets containing the keyword or hashtag.
    Returns a list of tweet texts.
    """
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    query = f"{keyword} -is:retweet lang:en"
    tweets = []
    try:
        for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=['text'], max_results=100).flatten(limit=max_tweets):
            tweets.append(tweet.text)
            time.sleep(1)  # Add a 1-second delay between tweets to avoid rate limits
    except tweepy.TooManyRequests:
        print("Rate limit exceeded. Please wait a few minutes and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return tweets 