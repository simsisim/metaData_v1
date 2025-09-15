import pandas as pd
import numpy as np

class GiustiScreener:
    def __init__(self, df):
        self.df = df

    def calculate_monthly_returns(self):
        return (self.df.pct_change() + 1).resample('ME').prod()

    def get_rolling_ret(self, df, n):
        return df.rolling(n).apply(np.prod)

    def get_top(self, date, ret_12, ret_6, ret_3):
        top_50 = ret_12.loc[date].nlargest(50).index
        top_30 = ret_6.loc[date, top_50].nlargest(30).index
        top_10 = ret_3.loc[date, top_30].nlargest(10).index
      
        return top_10

    def pf_performance(self, date, mtl, top_10):
        portfolio = mtl.loc['2024-10-31':, top_10][1:2]
        return portfolio.mean(axis=1).values[0]

    def run_screener(self):
        mtl = self.calculate_monthly_returns()
        ret_12 = self.get_rolling_ret(mtl, 12)
        ret_6 = self.get_rolling_ret(mtl, 6)
        ret_3 = self.get_rolling_ret(mtl, 3)
        results = []
        for date in mtl.index:
            top_10 = self.get_top(date, ret_12, ret_6, ret_3)
            performance = self.pf_performance(date, mtl, top_10)
            results.append({'Date': date, 'Top10': list(top_10), 'Performance': performance})

        return pd.DataFrame(results)

