import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')


def plot_star_rating_distribution(df):
    star_ratings = df['star_rating']

    # Set seaborn style
    sns.set_style("dark")

    # Create countplot
    plt.figure(figsize=(10, 6))
    sns.countplot(y=star_ratings, order=star_ratings.value_counts().sort_index(ascending=False).index , palette='Blues')
    plt.gca().set_facecolor('black')
    plt.xlabel('Count')
    plt.ylabel('Star Rating')
    plt.title('Distribution of Star Ratings')

    return plt

def plot_reviews_per_date(df):
    reviews_per_date = df.groupby('date').size().reset_index(name='Number of Reviews')

    sns.set_style("dark")

    plt.figure(figsize=(11,6))
    sns.lineplot(x='date', y='Number of Reviews', data=reviews_per_date, marker='o', color='skyblue')
    plt.gca().set_facecolor('black')
    plt.title('Number of Reviews Per Date')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=90)

    plt.tight_layout()

    return plt.gcf()
    
def plot_most_busy_moty(df):
    my = df['Month'].value_counts()
    sns.set_style('dark')
    plt.title("Most Busy Month of Year")
    sns.lineplot(x = my.index, y = my.values, color='#944dff')
    plt.gca().set_facecolor('black')
    plt.ylabel("Frequency")
    plt.xlabel("Day")
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    return plt

def analyze_sentiment(df):
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Tokenization and stopwords removal
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Check if the input is a string
        if isinstance(text, str):
            # Tokenize the text
            tokens = word_tokenize(text.lower())
        
            # Remove punctuation and stopwords
            tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
        
            return tokens
        else:
            return []

    # Function to perform sentiment analysis
    def calculate_sentiment(text):
        # Preprocess the text
        tokens = preprocess_text(text)
        
        # Calculate sentiment scores
        scores = sid.polarity_scores(' '.join(tokens))
        
        # Classify sentiment based on compound score
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
        
    # Apply sentiment analysis to each review body
    df['sentiment'] = df['review_body'].apply(calculate_sentiment)

    # Count the sentiment categories
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    # Extract counts for each sentiment category
    positive_count = sentiment_counts.get('positive', 0)
    negative_count = sentiment_counts.get('negative', 0)
    neutral_count = sentiment_counts.get('neutral', 0)

    # Calculate total number of sentiments
    total_count = positive_count + negative_count + neutral_count

    # Calculate the sentiment score out of 100
    if total_count > 0:
        score = ((positive_count - negative_count) / total_count) * 50 + 50
    else:
        score = 50  # Neutral if no sentiments

    # Convert counts to list of values and labels
    sentiment_labels = list(sentiment_counts.keys())
    sentiment_values = list(sentiment_counts.values())

    # Return the sentiment data and the score
    return {'labels': sentiment_labels, 'values': sentiment_values, 'score': score}

def plot_helpful_votes_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['helpful_votes'], kde=True, color='skyblue')
    plt.gca().set_facecolor('black')
    plt.xlabel('Number of Helpful Votes')
    plt.ylabel('Frequency')
    plt.title('Distribution of Helpful Votes')
    return plt

def plot_monthly_rating_trend(df):
    monthly_avg_rating = df.groupby(['Year', 'Month'])['star_rating'].mean()
    plt.figure(figsize=(10, 6))
    monthly_avg_rating.plot(marker='o', color='purple')
    plt.gca().set_facecolor('black')
    plt.xlabel('Month')
    plt.ylabel('Average Rating')
    plt.title('Monthly Average Rating Trend')
    return plt
def plot_top_positive_reviews(df, n=5,max_words=10):
    positive_reviews = df[df['sentiment'] == 'positive'].nlargest(n, 'helpful_votes')
    positive_reviews['review_body'] = positive_reviews['review_body'].apply(lambda x: ' '.join(x.split()[:max_words]))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='helpful_votes', y='review_body', data=positive_reviews, palette='Greens_r')
    plt.gca().set_facecolor('black')
    plt.xlabel('Helpful Votes')
    plt.ylabel('Review')
    plt.title('Top Positive Reviews')
    labels = positive_reviews['review_body'].tolist()
    values = positive_reviews['helpful_votes'].tolist()
    return {'labels': labels, 'values': values}

def plot_top_negative_reviews(df, n=5,max_words=10):
    negative_reviews = df[df['sentiment'] == 'negative'].nlargest(n, 'helpful_votes')
    negative_reviews['review_body'] = negative_reviews['review_body'].apply(lambda x: ' '.join(x.split()[:max_words]))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='helpful_votes', y='review_body', data=negative_reviews, palette='Reds_r')
    plt.gca().set_facecolor('black')
    plt.xlabel('Helpful Votes')
    plt.ylabel('Review')
    plt.title('Top Negative Reviews')
    labels = negative_reviews['review_body'].tolist()
    values = negative_reviews['helpful_votes'].tolist()
    return {'labels': labels, 'values': values}

def plot_positive_negative_reviews(df):
    positive_count = df[df['sentiment'] == 'positive'].shape[0]
    negative_count = df[df['sentiment'] == 'negative'].shape[0]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Positive', 'Negative'], y=[positive_count, negative_count], palette=['green', 'red'])
    plt.gca().set_facecolor('black')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Total Positive and Negative Reviews')
    
    # Add counts on top of the bars
    for i, v in enumerate([positive_count, negative_count]):
        plt.text(i, v, str(v), ha='center', va='bottom', color='white')
    
    return plt

def plot_positive_negative_reviews_pie(df):
    positive_count = df[df['sentiment'] == 'positive'].shape[0]
    negative_count = df[df['sentiment'] == 'negative'].shape[0]

    # Data for the pie chart
    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    colors = ['#76c7c0', '#ff6f61']  # Colors for the pie chart

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color':'white'})
    plt.gca().set_facecolor('black')
    plt.title('Distribution of Positive and Negative Reviews')

    return plt

def preprocess_data(df):
    # Check the columns to make sure they match
    expected_columns = ['review_title', 'reviewers_name', 'star_rating', 'review_date', 'helpful_votes', 'review_body']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("DataFrame does not have the expected columns. Please check the CSV file.")

    # Convert the 'review_date' to datetime and extract year, month, and day
    df['review_date'] = pd.to_datetime(df['review_date'], format='%d %B %Y')
    df['date'] = df['review_date'].dt.strftime('%d-%m-%Y')
    df['Year'] = df['review_date'].dt.year
    df['Month'] = df['review_date'].dt.month
    df['Day'] = df['review_date'].dt.day
    return df

# Function to handle plotting (example: star rating distribution)
def plot_star_rating_distribution(df):
    star_ratings = df['star_rating']

    sns.set_style("dark")

    plt.figure(figsize=(10, 6))
    sns.countplot(y=star_ratings, order=star_ratings.value_counts().sort_index(ascending=False).index, palette='Blues')
    plt.gca().set_facecolor('black')
    plt.xlabel('Count')
    plt.ylabel('Star Rating')
    plt.title('Distribution of Star Ratings')

    return plt


# def create_charts(data_path):
#     # Read data
#     df = pd.read_csv(data_path)
#     df = preprocess_data(df)

#     # Plot distribution of star ratings
#     plt = plot_star_rating_distribution(df)
#     plt2=plot_reviews_per_date(df)
#     plt3=plot_most_busy_moty(df)
#     plt4=analyze_sentiment(df)
#     plt5=plot_helpful_votes_distribution(df)
#     plt6=plot_monthly_rating_trend(df)
#     plt7=plot_top_positive_reviews(df)
#     plt8=plot_top_negative_reviews(df)
#     plt9=plot_positive_negative_reviews(df)
#     # Save the plot to a file
#     plt.savefig('star_rating_distribution.png')
#     plt2.savefig('reviews_per_date.png')
#     plt3.savefig('most_busy_moty.png')
#     plt4.savefig('sentiment.png')
#     plt5.savefig('helpful_votes.png')
#     plt6.savefig('monthly_ratings.png')
#     plt7.savefig("top_positive_reviews.png")
#     plt8.savefig("top_negative_reviews.png")
#     plt9.savefig("positive_negative_reviews.png")