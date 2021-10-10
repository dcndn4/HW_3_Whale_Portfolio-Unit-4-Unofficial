#### *Pandas_HW_3*
# Portfolio Analysis techniques with Python and Pandas

This file accompanies the new analysis notebook being provided to investment managers here at Company ABC. 

## Goal of This Notebook 
The goal of these analysis tools is to provide the ability to review portfolios consistently with useful analysis including:

> Volatility
> 
> Returns
> 
> Risk
> 
> Sharpe Ratios

As we start to look at portfolios with these tools, I'm sure we'll have lots of conversations about which are most useful. Included here are some starting points for those conversations.

### Volatility

Volatility is mainly captured within the annualized standard deviation, which multiplies the internal standard deviation (daily, weekly or monthly) by the square root of the number of intervals in the year (250, 52 or 12). That annualized standard deviation is usually in the range of 10% to 30%. Nasdaq's volatility (annualized standard deviation) is 28.8%, while the S&P 500's volatility is 18.1%.

An asset's volatility is "an annualized measure of dispersion in the stochastic process that is used to model the log returns." Most commonly modeled using          standard deviation (sigma). (~Carol Alexander, Practical Financial Econometrics, 90)
        
#### Standard Deviation

Standard deviation measures the distance between a set of values and their total average value.

Daily standard deviation is a parametric statistical measure, meaning it describes the data set within certain assumptions (such as mean distribution/ bell curve).  When a data set pattern is outside that usual distribution, a box plot can be utilized. Box plots contain information about the standard distribution of results, but also provide more meaningful information about all the rest of the data points. 
        
The standard distribution measures the dispersion of a dataset relative to its mean (i.e. average). It is calculated as the square root of variance: each data point's deviation from the mean. The greater that distance, the greater the standard deviation. 


Variance itself is an ingredient within standard deviation.. is usually not part of the analysis itself because it doesn't graph out in a way that is helpful (investopedia). Also, variance may end up being in a different unit of measurement than the data itself, which adds to the confusion.

##### Benchmarks for standard deviation: 
> Volatile stocks have high standard deviation

> 'Blue chip' stocks tend to have lower standard deviation
  
> Index funds are built in such a way as to have low volatility (lower standard deviation) in relation to their benchmark index - intended to replicate the results of the index.
  
> Standard deviation (std dev) is widely reported for any stock or investment instrument, by analysts, portfolio managers and advisors (and investors). 
  
> Standard deviation can be compared to stocks/funds in a specific asset class or market sector. Mostly, those instruments would be expected to be similar - as they're  affected by overall market conditions similarly. Stocks/funds within that group with higher volatility have something else going on (investopedia video).
          
##### Standard deviation usefulness:
is shown in the same unit of measurement as the underlying data, which helps make it understandable
graphical results are clear and easy to discuss
          
### Curve of standard deviation
normal curve - then 68% of results fall w/n 1 std dev, or mean, data point. 
shape with more volatility - fewer than 68% are within the 1 std dev, and more of the data points are outside of those bounds. 
          
##### Standard deviation downside: 
Standard deviation analysis doesn't supply a reason for the deviation.. could be the company experiences positive growth. That would be variance from the mean, so std dev is higher, so risk/reward are higher.. Kind of circuitious.. when reward is higher, risk is assumed to also be higher. Outliers and extreme values skew the shape of the curve, and may reduce the informational value of this calculation. When there are outliers, the normal distribution isn't applicable, but there isn't a different model to use instead basically, so talking about the results is more difficult. Past results don't predict the future, of course.

                
##### Primary relationship between standard deviation (1 vs. 2 etc..) and risk/reward. 
        
>   []  1 std dev may equate to being in the group of 70% of the total population of values
>   
>   []  2 std dev for instance might be within 95% of total population of the group
>   
>   []  3 std dev = 100% basically

           
Standard deviation is only helpful within context - since standard deviation is not relative measure, so has to have a context built around it. That context generally comes from similar funds, (meaning funds with same defining features) and/or a relevant index. For a single stock, context would come from other stocks from similar companies (market sector, size etc). 

## Returns

Returns are the change in the closing price of a stock or portfolio that day, compared to the prior day. 

### Rolling average or moving average - 

Rolling or moving average is a way to see things in perspective via increasing the signal-to-noise ratio.. removes large quantity of data points, and causes the remaining data points to have much more information. (per investopedia). The moving average erases the 'noise' aspect of information by muffling the random, short-term fluctuations. Instead each moving average data point is itself a summary of a period of time, and so with other similar points, paints a much more useful picture of what is going on. See 'rolling statistics' exercise for review. 

### Rolling Standard Deviation

Rather than look at a single standard deviation value for a stock, the rolling std dev is a graphical display of the cumulative std dev. For each point in the y axis (usually time), the rolling std dev is a value between 0 and 1 that reflects the std dev over a period of time prior to that point. The period of time (or window) in each case is provided. Common windows are 14 days or 20 days etc. The graph provides a boiled-down essence of the volatility of the stock or fund. 

### Application of Standard Deviation

Standard deviation is anchored in the past, and it's only useful regarding the future is the assumption is made that the past will be repeated. It is also called 'realized volatility' or 'historical votality (HV)', since it's based on past data.

The other way of predicting future volatility that isn't based on past results (like standard deviation), but is based on option prices. Publicly-known option prices carry information about investor expectations for prices in the future, based on expected volatility. That volatility when  distilled out from those assumptions (for a particular stock or fund) is called forward-looking "implied volatility(IV)". 

Standard deviation doesn't speak to returns directly.. so a small standard deviation in an area with low profitability with have lower returns that a portfolio with that same standard deviation in a sector with high profitability. Context is everything.


### Vix 
Vix is a volatility index. It's full name is the Cboe Volatility Index, abbreviated to VIX. It was created in 1993 by the Chicago Board Options Exchange (CBOE) and is maintained by Cboe global markets. It generates a 30-day forward projection of future volatility. It is constructed from the implied (future) volatilities of the S&P 500 - considered the leading indicator of the overall US Stock Market. Historical data demonstrates a strong negative correlation between volatility and stock market returns - when volatility rises, returns go down (and vice versa). (Investopedia).

### Covariance 

Covariance is a measure of dependency between two returns. Per Malkiel, it is the degree of parallelism between the returns of two securities. The calculation involves the actual and expected return of the two securities. Negative covariance means the two stocks move in opposite ways, they mitigate risk.

Another way to put it is that covariance measures the directional relationship between two variables. If two stocks go up at the same time, covariance is positive!
        
Choosing stocks with negative covariance is the essence of diversification, but doing that has become more and more difficult.

               
### Correlation -- hard-bound - varies between -1 and + 1
         
The higher the correlation, the less safety there is from market volatility
* with lower / negative  correlation, portfolio diversity is in place, which provides safety from market volatility

Correlations:
>      [] +1 = perfect positive relationship

>      []  0 = perfect random relationship

>      [] -1 = prefect negative relationship


### Beta - the specific volatility or systematic risk of a single stock, compared to the market as a whole

Beta is about the performance of a stock in relation to the volatility of the market. 

High-beta stocks are 'aggressive' investments, low-beta stocks are defensive. (- A RW d MS). 

A beta of 1 is assigned to a relevant market index. Then if a stock has a beta of 2, it loops out twice as far in every direction as the market index. If a stock has a beta of .5, it only moves half as far as the market. However it is crucial to carefully decide what to include in 'the market', as that determines the utility of beta as an investing tool.

Beta exists in a context provided by a benchmark (such as the s&p 500), and that benchmark needs to be relevant to that stock in order for the results to be useful. A numerical way to indicate the relevance of the benchmark to the stock is the R-squared value -- the higher that is, the more relevant the benchmark is. 

## Risk - Systematic Risk vs. Unsystematic risk

Systematic risk is an adverse pressure that affects the entire stock market - such as the financial crisis in 2008. Even very well-diversified portfolios were harmed by that event. 

Unsystematic risk or diversifiable risk is the uncertainty carried by a specific stock or fund or industry. That risk can be reduced via diversification. 

## Sharpe ratio - risk-adjusted investment portfolio analysis measure (from Wikipedia)

The Sharpe ratio was created by William Sharpe in 1966 (from Investopedia). It has been a popular reference point for investors ever since, even moreso since Professor Sharpe won a Nobel Memorial Prize in Economic Sciences in 1990 for his work on the Capital Asset Pricing Model (CAPM). He learned of the work of economist Harry Markowitz while at UCLA, and got his PhD under Markowitz guidance in 1961. He developed the Sharpe Ratio in 1966. 

The Sharpe ratio divides the profitability of a portfolio by its volatility, to give a measurement of the additional amount of return that an investor receives per unit of increase in risk.

Benchmarks for Sharpe Ratio: 1-2 is good, 2-3 is very good, and over 3 is excellent. 

# Technical Notes

## Libraries
This Jupyter Lab notebook utilizes the following libraries:

Pandas

Numpy

Datetime

Pathlib

matplotlib

## Data Inputs

The data input to this notebook is closing price data per datetime index. 
