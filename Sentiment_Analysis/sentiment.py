from textblob import TextBlob

def analyze_sentiment(text):
    """
    Returns: 'Positive', 'Neutral', or 'Negative'
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral' 