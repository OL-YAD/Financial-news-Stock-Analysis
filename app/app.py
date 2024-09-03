import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.sentiment_correlation_analysis import *

def load_data():
    data_directory = "../data/yfinance_data/" 
    stock_data = merge_stocks(data_directory)
    news_data = pd.read_csv('../data/raw_analyst_ratings.csv')
    
    # Data preprocessing
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    news_data['date'] = pd.to_datetime(news_data['date'], utc=True, format='mixed').dt.date
    news_data = perform_sentiment_analysis(news_data)
    news_data = news_data.rename(columns={'date': 'Date'})
    news_data = aggregate_daily_sentiment(news_data)
    stock_data = calculate_daily_returns(stock_data)
    
    # Merge datasets
    df = pd.merge(news_data, stock_data, on='Date', how='inner')
    return df

def main():
    st.title("Stock Sentiment Analysis Dashboard")

    # Load data
    df = load_data()

    # Sidebar for stock selection
    stocks = df['Stock'].unique()
    selected_stock = st.sidebar.selectbox("Select a stock", stocks)

    # Filter data for selected stock
    stock_data = df[df['Stock'] == selected_stock]

    # Display correlation
    st.header(f"Correlation Analysis for {selected_stock}")
    correlation = stock_data['Sentiment_Score'].corr(stock_data['Daily_Return'])
    st.write(f"Correlation between sentiment and daily returns: {correlation:.4f}")

    # Correlation strength interpretation
    def interpret_correlation(corr):
        if abs(corr) < 0.2:
            return "Very weak"
        elif abs(corr) < 0.4:
            return "Weak"
        elif abs(corr) < 0.6:
            return "Moderate"
        elif abs(corr) < 0.8:
            return "Strong"
        else:
            return "Very strong"

    strength = interpret_correlation(correlation)
    st.write(f"Correlation strength: {strength}")

    # Calculate p-value
    _, p_value = stats.pearsonr(stock_data['Sentiment_Score'], stock_data['Daily_Return'])
    st.write(f"P-value: {p_value:.4f}")

    # Visualizations
    st.header("Visualizations")

    # Time series plot
    st.subheader("Sentiment Score vs Daily Returns Over Time")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score', color='tab:blue')
    ax1.plot(stock_data['Date'], stock_data['Sentiment_Score'], color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Daily Return', color='tab:orange')
    ax2.plot(stock_data['Date'], stock_data['Daily_Return'], color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    plt.title(f'{selected_stock} - Sentiment Score vs Daily Returns')
    st.pyplot(fig)

    # Moving averages
    st.subheader("Moving Averages of Sentiment and Returns")
    stock_data['MA_Sentiment'] = stock_data['Sentiment_Score'].rolling(window=5).mean()
    stock_data['MA_Return'] = stock_data['Daily_Return'].rolling(window=5).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['Date'], stock_data['MA_Sentiment'], label='5-day MA Sentiment')
    ax.plot(stock_data['Date'], stock_data['MA_Return'], label='5-day MA Return')
    ax.set_title(f'{selected_stock} - Moving Averages of Sentiment and Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
