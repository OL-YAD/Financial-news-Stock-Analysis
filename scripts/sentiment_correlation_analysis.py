import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from textblob import TextBlob


# load stock data 
def load_stock_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    return df

# merge stocks data 
def merge_stocks(data_directory):
    stock_files = {
        'AAPL': 'AAPL_historical_data.csv',
        'AMZN': 'AMZN_historical_data.csv',
        'GOOGL': 'GOOG_historical_data.csv',
        'META': 'META_historical_data.csv',
        'MSFT': 'MSFT_historical_data.csv',
        'NVDA': 'NVDA_historical_data.csv',
        'TSLA': 'TSLA_historical_data.csv'
    }
    
    stock_data = pd.DataFrame()
    
    for ticker, file_name in stock_files.items():
        file_path = os.path.join(data_directory, file_name)
        if os.path.exists(file_path):
            df = load_stock_data(file_path)
            df['Stock'] = ticker
            stock_data = pd.concat([stock_data, df], ignore_index=True)
        else:
            print(f"File not found: {file_path}")
    
    # Sort the merged data by date and stock
    stock_data.sort_values(['Date', 'Stock'], inplace=True)
    
    return stock_data

# calculate sentiment
def calculate_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# calculate daily returns 
def calculate_daily_returns(df):
    df['Daily_Return'] = df.groupby('Stock')['Close'].pct_change()
    return df

# sentiment analysis on 'headline'
def perform_sentiment_analysis(df):
    df['Sentiment_Score'] = df['headline'].apply(calculate_sentiment)
    return df

# calculate mean daily sentiment 
def aggregate_daily_sentiment(df):
    return df.groupby('Date')['Sentiment_Score'].mean().reset_index()

# calculate correlation between Sentiment Score and Daily return
def calculate_correlation(df):
    """Calculate correlation between sentiment scores and stock returns."""
    correlations = df.groupby('Stock').apply(lambda x: x['Sentiment_Score'].corr(x['Daily_Return'], method='pearson'))
    return correlations

# tes correlation significance 
def test_correlation_significance(df, column1, column2):
    correlation, p_value = stats.pearsonr(df[column1], df[column2])
    return pd.Series({'correlation': correlation, 'p_value': p_value})
    correlations = df.groupby('Stock').apply(lambda x: x['Sentiment_Score'].corr(x['Lagged_Return']))
    return correlations