import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import  matplotlib.dates as mdates
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



# load data 
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# headline length
def headline_length(df):
    """Analyze headline length statistics."""
    df['headline_length'] = df['headline'].str.len()
    return df['headline_length'].describe()


#count number of articles per publisher 
def articles_per_publisher(df):
    return df['publisher'].value_counts()


# analyze publication dates 

def publication_dates(df):
    # Ensure 'date' is in datetime format
    #df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Group by date and count articles
    daily_counts = df.groupby(df['date'].dt.date).size()
    
    # Find days with highest article counts
    top_days = daily_counts.nlargest(5)
    
    # Analyze weekday distribution
    weekday_counts = df['date'].dt.day_name().value_counts()
    
    # Monthly trend
    df['month_start'] = df['date'].dt.floor('D') + MonthEnd(0) - MonthEnd(1)
    #monthly_counts = df.groupby('month_start').size()
    monthly_counts = df.groupby(df['date'].dt.to_period('M').dt.to_timestamp()).size()
    
    return {
        'daily_counts': daily_counts,
        'top_days': top_days,
        'weekday_counts': weekday_counts,
        'monthly_counts': monthly_counts
    }


#Plot the publication trends
def plot_publication_trends(date_analysis):

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Daily trend
    date_analysis['daily_counts'].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Daily Article Count')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Articles')
    
    # Top days
    date_analysis['top_days'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Top 5 Days with Most Articles')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Number of Articles')
    
    # Weekday distribution
    date_analysis['weekday_counts'].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Article Distribution by Weekday')
    axes[1, 0].set_xlabel('Weekday')
    axes[1, 0].set_ylabel('Number of Articles')
    
    # Monthly trend
    monthly_counts = date_analysis['monthly_counts']
    monthly_counts.plot(ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Article Count')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Articles')
    
    # Format x-axis to show months
    axes[1, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    
    plt.tight_layout()
    return fig


# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# sentiment analysis on the 'headline' column 
def perform_sentiment_analysis(df, text_column='headline'):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(x))
    df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))
    return df

