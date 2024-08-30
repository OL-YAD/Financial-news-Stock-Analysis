import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

def load_stock_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

# calculate technical indicators 
def technical_indicators(df):
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

# calculate financial metrics 
def financial_metrics(df):
    # Calculate daily returns using Adj Close
    df['Daily_Return'] = df['Adj Close'].pct_change()
    
    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    return df

def plot_stock_data(df, ticker):
    plt.figure(figsize=(12,8))
    plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    plt.plot(df.index, df['SMA_50'], label='50-day SMA')
    plt.title(f'{ticker} Adjusted Stock Price and 50-day SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_technical_indicators(df, ticker):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    ax1.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    ax1.set_title(f'{ticker} Technical Indicators')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    ax2.plot(df.index, df['RSI'], label='RSI')
    ax2.set_ylabel('RSI')
    ax2.legend()
    
    ax3.plot(df.index, df['MACD'], label='MACD')
    ax3.plot(df.index, df['MACD_Signal'], label='Signal Line')
    ax3.set_ylabel('MACD')
    ax3.legend()
    
    plt.xlabel('Date')
    plt.show()

# Financial analyzer 
def plot_financial_metrics(df, ticker):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    ax1.plot(df.index, df['Cumulative_Return'], label='Cumulative Return')
    ax1.set_title(f'{ticker} Financial Metrics')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    
    ax2.plot(df.index, df['Daily_Return'], label='Daily Return')
    ax2.set_ylabel('Daily Return')
    ax2.legend()
    
    ax3.plot(df.index, df['Volatility'], label='Volatility (20-day)')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    
    plt.xlabel('Date')
    plt.show()

# plot volume vs dividends
def plot_volume_and_dividends(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.bar(df.index, df['Volume'], label='Volume')
    ax1.set_title(f'{ticker} Volume and Dividends')
    ax1.set_ylabel('Volume')
    ax1.legend()
    
    ax2.bar(df.index, df['Dividends'], label='Dividends', color='green')
    ax2.set_ylabel('Dividends')
    ax2.legend()
    
    plt.xlabel('Date')
    plt.show()


# closing price time series analysis 
def plot_time_series_closing_price(df, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'])
    plt.title(f'Time Series of Closing Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()


#  Time Series Plot for Volume over  the years
def plot_time_series_volume(df, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Volume'])
    plt.title(f'Time Series of Trading Volume for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.show()


# Calculate and plot 30,60,90 moving averages for the closing price of the stock
def calculate_and_plot_moving_averages(df, ticker):
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA90'] = df['Close'].rolling(window=90).mean()

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.plot(df.index, df['MA30'], label='30-day MA')
    plt.plot(df.index, df['MA60'], label='60-day MA')
    plt.plot(df.index, df['MA90'], label='90-day MA')
    plt.title(f'{ticker} Closing Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# average closing price for the stock
def resample_and_plot_monthly_average(df, ticker):
    monthly_avg = df['Close'].resample('M').mean()
    plt.figure(figsize=(12,6))
    plt.plot(monthly_avg.index, monthly_avg)
    plt.title(f'{ticker} Monthly Average Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Average Closing Price')
    plt.show()

# Calculate daily price change percentage for the stock
def calculate_and_plot_daily_change(df, ticker):
    df['Daily_Change_Pct'] = df['Close'].pct_change() * 100
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Daily_Change_Pct'])
    plt.title(f'{ticker} Daily Price Change Percentage')
    plt.xlabel('Date')
    plt.ylabel('Price Change (%)')
    plt.show()

# calculate the day's price change percentage
def plot_volume_vs_price_change(df, ticker):
    df['Daily_Change_Pct'] = df['Close'].pct_change() * 100
    plt.figure(figsize=(12,6))
    plt.scatter(df['Daily_Change_Pct'], df['Volume'])
    plt.title(f'{ticker} Volume vs Price Change Percentage')
    plt.xlabel('Price Change (%)')
    plt.ylabel('Volume')
    plt.show()

# Seasonality check 
def resample_and_plot_monthly_average(df, ticker):
    # Add a Month column
    df['Month'] = df.index.month
    
    # Calculate average closing price for each month
    monthly_avg = df.groupby('Month')['Close'].mean()
    
    # Plot the monthly average closing price
    plt.figure(figsize=(14, 7))
    monthly_avg.plot(marker='o')
    plt.title(f'Average Monthly Closing Price for {ticker}')
    plt.xlabel('Month')
    plt.ylabel('Average Closing Price')
    plt.grid(True)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend([ticker])
    plt.tight_layout()
    plt.show()

    return monthly_avg