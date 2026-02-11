import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class MinerviniScreener:
    def __init__(self, df, rs_results, model_params):
        self.df = df
        self.rs_results = rs_results
        self.benchmark_ticker = model_params['benchmark_ticker']

    def calculate_sma(self, period):
        return self.df.rolling(window=period).mean()

    def calculate_ema(self, period):
        return self.df.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, period):
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

        
        # Extract RS values for the given ticker
        #rs_values = self.rs_results.loc[self.rs_results['Symbol'] == ticker, rs_columns].values[0]
        
        # Calculate weighted average using provided weights
        #r#eturn np.dot(rs_values, self.weights)

    def run_screener(self):
        sma150 = self.calculate_sma(150)
        sma200 = self.calculate_sma(200)
        ema50 = self.calculate_ema(50)
        
        

        results = []
        
        for ticker in self.df.columns:
            if ticker == self.benchmark_ticker:  # Skip the benchmark ticker
                continue
            stock_data = self.df[ticker]
            current_close = stock_data.iloc[-1]

            # Criterion 1: Current price > 150 SMA and > 200 SMA
            criterion1 = min(current_close / sma150[ticker].iloc[-1], current_close / sma200[ticker].iloc[-1])
            #print(criterion1)

            # Criterion 2: 150 SMA > 200 SMA
            criterion2 = sma150[ticker].iloc[-1] / sma200[ticker].iloc[-1]

            # Criterion 3: 200 SMA trending up for at least 1 month
            criterion3 = sma200[ticker].iloc[-1] / sma200[ticker].iloc[-22]

            # Criterion 4: 50 EMA > 150 SMA and 50 EMA > 200 SMA
            criterion4 = min(ema50[ticker].iloc[-1] / sma150[ticker].iloc[-1], ema50[ticker].iloc[-1] / sma200[ticker].iloc[-1])

            # Criterion 5: Current price > 50 EMA
            criterion5 = current_close / ema50[ticker].iloc[-1]

            # Criterion 6: Current price at least 30% above 52-week low
            low_52week = stock_data.rolling(window=252).min().iloc[-1]
            criterion6 = current_close / (1.3 * low_52week)

            # Criterion 7: Current price within 25% of 52-week high
            high_52week = stock_data.rolling(window=252).max().iloc[-1]
            criterion7 = current_close / (0.75 * high_52week)

            # Criterion 8: RS Rating Weighted Average > 80
            rs_row = self.rs_results.loc[self.rs_results['Symbol'] == ticker].iloc[0]
            #print('index', rs_row.index)
            #print(rs_row)
            rs_rating_wa = self.rs_results.loc[self.rs_results['Symbol'] == ticker, 'RS_Rating_WA'].values[0]
            #print(criterion1, rs_rating_wa)
            if all([
                criterion1 > 1,
                criterion2 > 1,
                criterion3 > 1,
                criterion4 > 1,
                criterion5 > 1,
                criterion6 >= 1,
                criterion7 >= 1,
                rs_rating_wa > 70
            ]):
                result_dict = {
                    'Symbol': ticker,
                    'Price': round(current_close, 2),
                    'SMA150': round(sma150[ticker].iloc[-1], 2),
                    'SMA200': round(sma200[ticker].iloc[-1], 2),
                    'EMA50': round(ema50[ticker].iloc[-1], 2),
                    'Criterion1': round(criterion1, 2),
                    'Criterion2': round(criterion2, 2),
                    'Criterion3': round(criterion3, 2),
                    'Criterion4': round(criterion4, 2),
                    'Criterion5': round(criterion5, 2),
                    'Criterion6': round(criterion6, 2),
                    'Criterion7': round(criterion7, 2),
                    'Criterion8': rs_rating_wa,
                }
                # Add rs_results columns dynamically
                for col in rs_row.index:
                    if col != 'Symbol':  # Assuming 'Symbol' is already included
                        result_dict[col] = rs_row[col]
            
                results.append(result_dict)
        columns = ['Symbol', 'Price', 'SMA150', 'SMA200', 'EMA50', 'Criterion1', 'Criterion2', 
               'Criterion3', 'Criterion4', 'Criterion5', 'Criterion6', 'Criterion7', 'Criterion8']
        # Dynamically generate columns list for rs_results
        #print(results)
        if results:
            rs_columns = list(results[0].keys())
            columns = [col for col in columns if col in rs_columns] + [col for col in rs_columns if col not in columns]
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=columns)




