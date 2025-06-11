# For division in Python 3
from __future__ import division

# pandas and NumPy imports
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime

# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# For loop for grabbing yahoo finance data and setting as a dataframe
for stock in tech_list:   
    # Set DataFrame as the Stock Ticker
    globals()[stock] = DataReader(stock, 'yahoo', start, end)

# Summary Stats for Apple stocks
AAPL.describe()

# General Info about Apple Stock
AAPL.info()

# Historical view of the closing price of Apple stock
AAPL['Adj Close'].plot(legend=True, figsize=(10, 4))

# Historical view of the total volume of Apple stock traded each day
AAPL['Volume'].plot(legend=True, figsize=(10, 4))

# Calculation of moving averages for 10, 20 and 50 days of Apple stocks
ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = "MA for %s days" % (str(ma))
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma).mean()

# Historical view of the moving averages of Closing Price of Apple Stock
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize=(10, 4))

# Calculation to find the percent change for each day of Apple stock
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

# Visualization of the percent change for each day of Apple stock
AAPL['Daily Return'].plot(figsize=(12, 4), legend=True, linestyle='--', marker='o')

# Histogram to visualize the average daily return of Apple stock
sns.histplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')

# Calculation to grab all the closing prices for the tech stock list into one DataFrame
closing_df = DataReader(['AAPL', 'GOOG', 'MSFT', 'AMZN'], 'yahoo', start, end)['Adj Close']

# Quick look of the data frame
closing_df.head()

# Calculate the daily return percent of all stocks and store them in a new tech returns DataFrame
tech_rets = closing_df.pct_change()

# Comparing Google to itself shows a perfectly linear relationship
sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')

# Joinplot to compare the daily returns of Google and Microsoft
sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')

# Correlation analysis for every possible combination of stocks in our technology stock ticker list.
sns.pairplot(tech_rets.dropna())

# Mixed plot to visualize the correlation between all technology stocks
returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

# Correlation analysis by using mixed types of plots for the closing price of all technology stocks
returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

# Correlation plot for the daily returns of all stocks
sns.heatmap(tech_rets.corr(), annot=True, cmap='coolwarm')

# There are many ways we can quantify risk, one of the ways is by comparing the expected return with the standard deviation of the daily returns.
# Cleaning data frame by dropping rows having null values
rets = tech_rets.dropna()

# Scatter plot of expected return of the stocks vs. their standard deviations of daily returns
area = np.pi * 20

plt.scatter(rets.mean(), rets.std(), alpha=0.5, s=area)
plt.ylim([0.01, 0.025])
plt.xlim([-0.003, 0.004])
plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy=(x, y), xytext=(50, 30),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.5'))
        
plt.show()
