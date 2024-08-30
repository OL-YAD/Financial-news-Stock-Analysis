# Financial-news-Stock-Analysis

## Project Overview
This project is focused on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements. 

### Main Objectives
1. Perform sentiment analysis on financial news headlines.
2. Establish statistical correlations between news sentiment and stock price movements.
3. Provide actionable insights and investment strategies based on your analysis.

## Folder Structure
- `notebooks/`: Jupyter notebooks for EDA.
       -  `Sentiment_Analysis_EDA.ipynb` : EDA analysis for Financial News Analysis 
- `scripts/`: Python scripts for EDA processing.

## Directory Structure

```plaintext
├── .vscode/
│   └── settings.json          
├── .github/
│   └── workflows/
│       └── unittests.yml      # GitHub Actions
├── .gitignore                 # directories to be ignored by git
├── requirements.txt           # contains dependencies for the project
├── README.md                  # Project documentation (this file)
├── src/
│   ├── __init__.py
│   
├── notebooks/
│   ├── __init__.py
│   ├── Sentiment_Analysis_EDA.ipynb  # Jupyter notebook for stock news EDA analysis
│   └── AAPL_EDA.ipynb,AMZN_EDA.ipynb,GOOG_EDA.ipynb,META_EDA.ipynb,MSFT_EDA.ipynb,NVDA_EDA.ipynb, TSLA_EDA ipynb          #  notebook files for financial analysis of each stock data 
├── tests/
└── scripts/
    ├── __init__.py
    ├── utils.py # Script for financial news analysis 
    ├── financial_analyzer.py # script for the stock data analysis    
    └── README.md             # Documentation for the scripts directory
```
## Setup Instructions
1. Clone the repository.
2. Set up the virtual environment.
3. Install dependencies using `pip install -r requirements.txt`