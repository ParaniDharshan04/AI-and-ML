import matplotlib.pyplot as plt

def plot_pie(sentiment_counts):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['green', 'red', 'blue']
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Sentiment Distribution')
    plt.show()

def plot_bar(sentiment_counts):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['green', 'red', 'blue']
    plt.figure(figsize=(6,4))
    plt.bar(labels, sizes, color=colors)
    plt.title('Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.show() 