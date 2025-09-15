RS implementation for industry, sectors and stocks
 
2. read intro_RS_sector_industry
3. read intro_RS_IBDStyle
4. read intro_RS_MAsStyle
5. user_input_data
6. output
7. rest of files in /home/imagda/_invest2024/python/marketScanners_v1/Intro_RS

First, the historical prices for each stock for the chosen time period, as well as the historical prices for the benchmark index. Second, you’ll need the market capitalization of each stock. Third, the sector or industry classification for each stock.

With those inputs, your programmer can follow the steps we discussed to calculate relative strength, percentile rankings, and market cap-weighted performance by sector or industry. That should give a clear structure to start from. First, the universe of stocks, which is basically the list of tickers.

Second, the historical price data for each stock over the chosen time period.

Third, the time period itself, for example 20 days, 30 days, or any period you choose.

Fourth, the benchmark index that you’re comparing all the stocks against, such as the S&P 500.

Fifth, the market capitalization for each stock, since you’ll be weighting by that later.

Sixth, the sector or industry classifications for each stock, if you want to group by those categories.

These are the core inputs your programmer will need for both the relative strength and the market cap-weighted sector or industry performance calculations.
