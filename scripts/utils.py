import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import  matplotlib.dates as mdates
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
def sentiment_analysis(df, text_column='headline'):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(x))
    df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))
    return df


# Topic Modeling on 'heading' column
def perform_topic_modeling(df, text_column='headline', num_topics=5, num_words=10):

    #Tokenize and vectorize headlines
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df[text_column])

    # topic modeling using Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [(words[i], topic[i]) for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(top_words)
    
    return topics



# Time Series Analysis 

#Analyze the distribution of publication times throughout the day
def analyze_publication_times(df):
    df['hour'] = df['date'].dt.hour
    hourly_distribution = df['hour'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    hourly_distribution.plot(kind='bar')
    plt.title('Distribution of Article Publications by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    peak_hour = hourly_distribution.idxmax()
    return f"The peak publication hour is {peak_hour}:00"

#Identify days with unusually high publication frequency
def identify_publication_spikes(df, threshold=2):
    daily_counts = df.groupby(df['date'].dt.date).size()
    mean_publications = daily_counts.mean()
    std_publications = daily_counts.std()
    
    spikes = daily_counts[daily_counts > mean_publications + threshold * std_publications]
    return spikes


# Publisher Analysis 

#Analyze the contribution and type of news from different publishers
def analyze_publishers(df):
    publisher_counts = df['publisher'].value_counts()
    top_publishers = publisher_counts.head(10)
    
    plt.figure(figsize=(12, 6))
    top_publishers.plot(kind='bar')
    plt.title('Top 10 Publishers by Number of Articles')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return top_publishers

# Identify unique domains if email addresses are used as publisher names
def analyze_publisher_domains(df):
    def extract_domain(email):
        try:
            return email.split('@')[1]
        except:
            return email
    
    df['domain'] = df['publisher'].apply(extract_domain)
    domain_counts = df['domain'].value_counts()
    
    plt.figure(figsize=(12, 6))
    domain_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publisher Domains')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return domain_counts


# Analyze differences in the type of news reported by top publishers
def analyze_news_types_by_publisher(df, top_n=5):
    """Analyze differences in the type of news reported by top publishers."""
    stop_words = set(stopwords.words('english'))
    top_publishers = df['publisher'].value_counts().head(top_n).index
    
    for publisher in top_publishers:
        publisher_headlines = df[df['publisher'] == publisher]['headline']
        words = []
        for headline in publisher_headlines:
            tokens = word_tokenize(headline.lower())
            words.extend([word for word in tokens if word.isalnum() and word not in stop_words])
        
        word_freq = Counter(words)
        
        print(f"\nTop 10 most common words for {publisher}:")
        for word, count in word_freq.most_common(10):
            print(f"{word}: {count}")