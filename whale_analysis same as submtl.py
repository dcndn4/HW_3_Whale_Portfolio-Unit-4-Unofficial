# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:51:55 2021

@author: CS_Knit_tinK_SC
"""

# Initial imports
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

%matplotlib inline


#%%

#file_path = Path="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/loans.csv"
#csv_path = Path="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/goog_google_finance.csv"
csv_path_w = "C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/Whale/whale_returns.csv"
whale_returns = pd.read_csv(
    csv_path_w, parse_dates=True, index_col="Date", infer_datetime_format=True
)

# whale df is 1060/4
print(whale_returns.head())
#%%

# Count nulls
print(whale_returns.isnull().mean() * 100)

#%%

whale_returns=whale_returns.dropna()


#%%

#file_path = Path="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/loans.csv"
#csv_path = Path="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/goog_google_finance.csv"
csv_path_a = "C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/Whale/algo_returns.csv"
algo_returns = pd.read_csv(
    csv_path_a, parse_dates=True, index_col="Date", infer_datetime_format=True
)

# algo df is 1241/2
# Replaced space with underscore in column headings per best functionality
algo_returns.columns=["Algo_1", "Algo_2"]
print(algo_returns.head())

#%%

# Count nulls
print(algo_returns.isnull().mean() * 100)

#%%

algo_returns=algo_returns.dropna()

#%%

# S&P 500 Returns

# Read the S&P 500 historic closing prices and create a new daily returns DataFrame from the data.


#file_path = Path="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/loans.csv"
#csv_path = Path="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/goog_google_finance.csv"
csv_path_s = "C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/Whale/sp500_history.csv"
#sp500_history = pd.read_csv(
#csv_path_s, parse_dates=True, index_col="Date", infer_datetime_format=True
#)
sp500_history = pd.read_csv(csv_path_s)

# sp500 is 1649/2
print(sp500_history.head())

#%%

# Check Data Types
print(sp500_history.dtypes)

#%%

#Set index as date field

#sp500_history.set_index(sp500_history['Date'], inplace=True)
#print(sp500_history.head())
sp500_history.set_index(pd.to_datetime(sp500_history['Date'], infer_datetime_format=True), inplace=True)
sp500_history.sort_index(inplace=True)
print(sp500_history.head())


#%%

# Fix Data Types


sp500_history['Close'] = sp500_history['Close'].str.replace('$','')
sp500_history['Close']=sp500_history['Close'].astype('float')


#%%
# Drop the extra date column
sp500_history.drop(columns=['Date'], inplace=True)
print(sp500_history.head())
print(sp500_history.dtypes)

#%%

sp500_history=sp500_history.dropna()

#%%

sp500_returns = sp500_history.pct_change()

#%%

sp500_returns.columns = ["SP500"] 

#%%

# Combine Whale, Algorithmic, and S&P 500 Returns


all_returns = pd.concat([algo_returns, whale_returns, sp500_returns], axis='columns', join = 'inner')

#%%
#%%
#%%

# Conduct Quantitative Analysis

# In this section, you will calculate and visualize performance and risk metrics for the portfolios.
# Performance Anlysis
# Calculate and Plot the daily returns.


# Plot daily returns of all portfolios
algo_returns.plot(title="Daily Returns")
whale_returns.plot(title="Daily Returns")
sp500_returns.plot(title="Daily Returns")

#%%

cumulative_algo_returns = (1 + algo_returns).cumprod() - 1
cumulative_whale_returns = (1 + whale_returns).cumprod() - 1
cumulative_sp500_returns = (1 + sp500_returns).cumprod() - 1

#%%

cumulative_algo_returns.plot(title="Cumulative Returns")
cumulative_whale_returns.plot(title="Cumulative Returns")
cumulative_sp500_returns.plot(title="Cumulative Returns")


#%%
#%%

# Risk Analysis

# Determine the risk of each portfolio:

# Create a box plot for each portfolio.
# Calculate the standard deviation for all portfolios
# Determine which portfolios are riskier than the S&P 500
# Calculate the Annualized Standard Deviation

# Create a box plot for each portfolio


all_returns.plot.box(notch='True', vert = 0, title="Risk per Portfolio Box Plot")

#%%

# Calculate the daily standard deviations of all portfolios

daily_std = all_returns.std()
print(f' The standard deviations for each portfolio are: \n{daily_std}')

#%%

# Calculate  the daily standard deviation of S&P 500

print(f' The daily Standard deviations are: \n{daily_std}')

# Calculate  the daily standard deviation of S&P 500

daily_std = daily_std.sort_values(ascending=False)

print(f' The sorted daily Standard deviations are: \n{daily_std}')
print(f'Berkshire Hathaway (.012919) and Tiger Global (.010894) are riskier than the S&P 500')



#%%

# Calculate the annualized standard deviation (252 trading days)
annualized_std = daily_std * np.sqrt(252)

print(f' The annualized standard deviations for each portfolio are: \n{annualized_std}')

#%%
#%%

#Rolling Statistics

#Risk changes over time. Analyze the rolling statistics for Risk and Beta.

#Calculate and plot the rolling standard deviation for all portfolios
# using a 21-day window
#Calculate the correlation between each stock to determine which 
#portfolios may mimic the S&P 500
#Choose one portfolio, then calculate and plot the 60-day rolling beta
# between it and the S&P 500



#%%

# Calculate and plot rolling std for all portfolios with 21-day window

# Calculate the rolling standard deviation for all portfolios using a 21-day window


#all_returns.rolling(window=21).std()   

# Plot the rolling standard deviation
#all_returns.rolling(window=21).std().plot

# Plot the rolling standard deviation using 21-day window
algo_returns.rolling(window=21).std().plot(title="Rolling Standard Deviation (21-day window)")
whale_returns.rolling(window=21).std().plot(title="Rolling Standard Deviation (21-day window)")
sp500_returns.rolling(window=21).std().plot(title="Rolling Standard Deviation (21-day window)")
#%%

# Calculate the correlation
correlation = all_returns.corr()
# Display the correlation matrix
print(correlation)
correlation.plot(title="Portfolio Correlations")

#%%

# Calculate covariance of a single portfolio
covariance = all_returns['Algo_1'].cov(all_returns['SP500'])

# Calculate variance of S&P 500
variance = all_returns['SP500'].var()

# Computing beta
Algo1_beta = covariance/variance

#%%

# Plot beta trend
rolling_Algo1_covariance = all_returns['Algo 1'].rolling(window=30).cov(all_returns['SP500'])
rolling_variance = all_returns['SP500'].rolling(window=30).var()
rolling_Algo1_beta = rolling_Algo1_covariance / rolling_variance
ax = rolling_Algo1_beta.plot(figsize=(20, 10), title='Rolling 30-Day Beta of Algo 1')

#%%

# Rolling Statistics Challenge: Exponentially Weighted Average

# An alternative way to calculate a rolling window is to take the exponentially weighted moving average. 
# This is like a moving window average, but it assigns greater importance to more recent observations. 
# Try calculating the ewm with a 21-day half-life.

#%%

# ewm some more: 
    
# code from P4DA:
#aapl_px = close_px.AAPL['2006' : '2007']
#ma60 = aapl_px.rolling(30, min_periods=20).mean()
#ewma60 = aapl_px.ewm(span=30).mean()
#ma60.plot(style='k--', label='Simple MA')
#ewma60.plot(style='k-', label='EW MA')

Algo1_px = all_returns.Algo_1['2019' : '2021']


ma21 = Algo1_px.rolling(21, min_periods=7).mean()


ewma21 = Algo1_px.ewm(span=21).mean()


ma21.plot(style='k--', label='Simple MA')


ewma21.plot(style='k-', label='EW MA', title="Rolling Window via EWM")

#%%

# Sharpe Ratios

# In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, 
# and not just returns alone. 
# After all, if you could invest in one of two portfolios, and each offered the same 10% return, 
# yet one offered lower risk, you'd take that one, right?

# Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

#%%

# Sharpe ratio
# some reference examples:
   
# From Sharpe Ratio demo
# sharpe_ratios =  Numerator: ((all_portfolios_returns.mean()-all_portfolios_returns['rf_rate'].mean()) * 252) 
# /
#   Denominator:  (all_portfolios_returns.std() * np.sqrt(252))

# From Risky Business demo
# sharpe_ratios = Numerator: (all_returns.mean() * 252) 
# / 
#   Denominator:  (all_portfolio_std * np.sqrt(252))

#print(sharpe_ratios.head())

print(f'S.R. num = {all_returns.mean() * 252}')
print(f'S.R. den = {daily_std * np.sqrt(252)}')

sharpe_ratios = (all_returns.mean() * 252)/(daily_std * np.sqrt(252))

print(f' Sharpe Ratios: \n{sharpe_ratios} \n Note: currently risk-free return is numerically similar to zero, so not included. \n')
#%%

# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot(kind="bar", title="Portfolio Sharpe Ratios")

#%%
#%%
#%%

#Create Custom Portfolio

#In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500.

#    Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
#    Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
#    Join your portfolio returns to the DataFrame that contains all of the portfolio returns
#    Re-run the performance and risk analysis with your portfolio to see how it compares to the others
#    Include correlation analysis to determine which stocks (if any) are correlated

#Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.

#For this demo solution, we fetch data from three companies listes in the S&P 500 index.

#    GOOG - Google, LLC

#    AAPL - Apple Inc.

#    COST - Costco Wholesale Corporation

#%%

# Reading data from 1st stock 
csv_path_g = "C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/Whale/yr_goog15.csv"
yr_goog = pd.read_csv(
    csv_path_g, parse_dates=True, index_col="Date", infer_datetime_format=True
)
print(yr_goog.head)



yr_goog=yr_goog.dropna()
yr_goog.columns = ["Google"] 
print(yr_goog.head)

#%%

# Reading data from 2nd stock
csv_path_a = "C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/Whale/yr_aapl15.csv"
yr_aapl = pd.read_csv(
    csv_path_a, parse_dates=True, index_col="Date", infer_datetime_format=True
)
print(yr_aapl.head)

yr_aapl=yr_aapl.dropna()
yr_aapl.columns = ["Apple"] 
print(yr_aapl.head)

#%%

# Reading data from 3rd stock
csv_path_c = "C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/Whale/yr_cost15.csv"
yr_cost = pd.read_csv(
    csv_path_c, parse_dates=True, index_col="Date", infer_datetime_format=True
)
print(yr_cost.head)
yr_cost=yr_cost.dropna()
yr_cost.columns = ["CostCo"] 
print(yr_cost.head)

#%%

# Combine all stocks in a single DataFrame
yr_all = pd.concat([yr_goog, yr_aapl, yr_cost], axis='columns', join = 'inner')
#%%

# sp500_history.set_index(pd.to_datetime(sp500_history['Date'], infer_datetime_format=True), inplace=True)

#%%


# Reorganize portfolio data by having a column per symbol
#yr_all.columns=["Google", "Apple", "CostCo"]
#%%

# Calculate daily returns
yr_all_returns = yr_all.pct_change()
#%%

# Drop NAs
yr_all_returns=yr_all_returns.dropna()
yr_all_returns.index=yr_all_returns.index.date
# Display sample data
print(yr_all_returns.head)

#%%

#Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock¶

# Set weights
weights = [1/3, 1/3, 1/3]

goog_weight = 0.33
aapl_weight = 0.33
cost_weight = .34
# Calculate portfolio return

yr_wt_returns = goog_weight * yr_all_returns["Google"] + aapl_weight * yr_all_returns["Apple"] + cost_weight * yr_all_returns["CostCo"]
#yr_wt_returns.rename(), = ["Big_3"] 

print(yr_wt_returns.head)
#%%

#yr_wt_returns.append(yr_wt_returns.sum().rename('Total'))

#%%

#Join your portfolio returns to the DataFrame that contains all of the portfolio returns

# change index of all_returns dateframe to be date only


all_returns.index=all_returns.index.date
print(all_returns.head)

#%%

#check data types)
print(all_returns.dtypes) 
print(yr_all_returns.dtypes)

#%%
total_returns = pd.concat([all_returns, yr_wt_returns], axis='columns', join = 'inner')
total_returns.columns=['Algo_1', 'Algo_2', 'Fund_Mtmg', 'Paulson', 'Global', 'BH', 'SP500', 'Big_3']

print(total_returns.head())
#%%
print(total_returns.dtypes)

#%%
# Only compare dates where return data exists for all the stocks (drop NaNs)
total_returns=total_returns.dropna()

#%%

# Re-run the risk analysis with your portfolio to see how it compares to the others

# Calculate the Annualized Standard Deviation

# Calculate the annualized `std` (252 trading days, same as earlier)

# Calculate the annualized standard deviation (252 trading days)

total_daily_std = total_returns.std()
print(f' Total Daily Standard Deviations are: \n{total_daily_std}')

#%%



# Sort  the daily standard deviation results

total_daily_std = total_daily_std.sort_values(ascending=False)
print(f' Sorted Total Daily Standard Deviations are: \n{total_daily_std}')

#%%



# Determine which portfolios are riskier than the S&P 500
# do for loop, identify items in riskier file??
total_annualized_std = total_daily_std * np.sqrt(252)
total_annualized_std = total_annualized_std.sort_values(ascending=False)
print(total_annualized_std)
print(f' Sorted Total Annualized Standard Deviations are: \n{total_annualized_std}')
#%%
# not asked for

# annualized_total_returns = total_returns * np.sqrt(252)

# print(annualized_total_returns)
#%%

# Calculate and plot rolling std with 21-day window

# Calculate rolling standard deviation

# Plot rolling standard deviation

# Plot the rolling standard deviation using 21-day window
total_returns.rolling(window=21).std().plot()

#%%

# Calculate and plot the correlation

# Calculate and plot the correlation

T_correlation = total_returns.corr()
# Display the correlation matrix
T_correlation
T_correlation.plot()

#%%

# Calculate and Plot Rolling 60-day Beta for Your Portfolio compared to the S&P 500

#%%

# Calculate covariance of a single portfolio
covariance = total_returns['Big_3'].cov(all_returns['SP500'])
print(covariance)


#%%

# Calculate variance of S&P 500
variance = all_returns['SP500'].var()

# Computing beta
Big_3_beta = covariance/variance
print(Big_3_beta)

#%%


# Plot beta trend
rolling_Big_3_covariance = total_returns['Big_3'].rolling(window=30).cov(all_returns['SP500'])
rolling_variance = all_returns['SP500'].rolling(window=30).var()
rolling_Big_3_beta = rolling_Big_3_covariance / rolling_variance
ax = rolling_Big_3_beta.plot(figsize=(20, 10), title='Rolling 30-Day Beta of Big_3')

#%%

# Calculate Annualzied Sharpe Ratios

# Annualized Sharpe Ratios

# Calculate sharpe ratio
#sharpe_ratios = (total_returns.mean() * 252) / (all_portfolio_std * np.sqrt(252))
#sharpe_ratios.head()

#annualized_total_returns = total_returns * np.sqrt(252) -- part done already
# this section is in case new person takes over, and wants to review components of Sharpe Ratios more closely.

#print(f'S.R. num = {total_returns.mean() * 252}')
print(f' These are the Sharpe Ratio numerator values: \n{total_returns.mean() * 252} \n Note: currently risk-free return is numerically similar to zero, so not included. \n')
#print(f'S.R. den = {total_annualized_std}')
print(f' These are the Sharpe Ratio denominator values: \n{total_annualized_std}')
#%%


total_sharpe_ratios = (total_returns.mean() * 252)/(total_annualized_std)

print(f' Total Sharpe Ratios are: \n{total_sharpe_ratios}')
#%%

# Visualize the sharpe ratios as a bar plot
total_sharpe_ratios.plot(kind="bar", title="Sharpe Ratios")

#%%

# How does your portfolio do?

# Write your answer here!


# My portfolio - The Big 3 - of Apple, Google and Costco performs well. 
# Its Sharpe ratios are inbetween that of the algorithmic-1 portfolio and the algorithmic-2 portfolio.


