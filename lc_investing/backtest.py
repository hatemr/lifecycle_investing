from copy import deepcopy
import cpi
import datetime
from dateutil import relativedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import os
import sys
from time import strptime
from typing import List
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
# sys.path.append('..')

class Backtest:
    """backtest class"""

    def __init__(self,
                 initial_leverage: float = 1):
        self.initial_leverage = initial_leverage

    def get_data(self,
                 start_date: str,
                 end_date: str,
                 data_filename: str = "../Core_simulation.xlsx"):
        """get data"""
        print()
        data = (
            pd.read_excel(data_filename,
                          sheet_name='Monthly data',
                          header=0)
            # assign dates using last day of the month
            .assign(date = lambda data: pd.to_datetime(data.loc[:, 'Monthly years'].astype(str), format="%Y.%m") + MonthEnd(0))
            .iloc[0:1662, :]  # remove last data points since they aren't accurate
            # filter to desired dates
            .query(f"date >= '{start_date}'")
            .query(f"date <= '{end_date}'")
            .reset_index(drop=True)
            # .loc[:, ['date', 'Monthly real stock rate',
            #          'Monthly real gov bond rate',
            #          'Monthly real margin rate']]
            .loc[:, ['date',
                     'Monthly nom stock return**',
                     'Monthly nom gov bond rate*',
                     'Monthly nom margin rate†']]
            .rename(columns={'Monthly nom stock return**': 'SP500',
                             'Monthly nom gov bond rate*': '10Ybond',
                             'Monthly nom margin rate†': 'margin_rate'})
            .melt(id_vars='date', var_name='id')
            .rename(columns={'value': 'returns'})
        )
        trend_socgen, trend_barclays = self.get_trend_index_data()
        index_data = self.get_index_data()

        data1 = pd.concat([data, trend_socgen, trend_barclays, index_data], axis=0)
        # convert from nominal to real
        data2 = pd.concat([adjust_to_real(data.query(f"id=='{id}'")) for id in data['id'].unique()], axis=0)
        
        # change simple returns R to gross returns (1+R)
        data1.loc[:, ['Monthly real stock rate', 'Monthly real gov bond rate', 'Monthly real margin rate']] = (
            1 + data1.loc[:, ['Monthly real stock rate', 'Monthly real gov bond rate', 'Monthly real margin rate']]
        )
        # turns the borrowing rate to zero - ad hoc
        # data1.loc[:, 'Monthly real margin rate'] = 1.0
        # initialize asset class values
        data1.loc[0, 'equity_value'] = self.initial_leverage
        data1.loc[0, 'bonds_value'] = 0
        data1.loc[0, 'debt_value'] = 1 - self.initial_leverage
        data1.loc[0, 'port_value'] = data1.loc[0, ['equity_value', 'bonds_value', 'debt_value']].sum()
        assert data1.loc[0, 'port_value'] == 1, "debts must equal assets"
        # initialize leverage
        data1.loc[0, 'leverage'] = (
                (data1.loc[0, ['equity_value', 'bonds_value']].sum())
                / data1.loc[0, 'port_value']
        )
        # initialize portfolio returns
        data1.loc[0, 'returns_port'] = np.nan
        # initialize cumulative portfolio returns
        data1.loc[0, 'returns_port_cum'] = 1
        data1.loc[0, ['weight_equity', 'weight_bond', 'weight_debt']] = (
            data1.loc[0, ['equity_value', 'bonds_value', 'debt_value']].values
            / data1.loc[0, 'port_value']
        )

        self.data=data1

    def run(self,
            rebalance_dates: list):
        """run the backtest"""
        data = deepcopy(self.data)   

        # run the backtest
        for i, row in data.iloc[1:, :].iterrows():
            # print(i, row['date'])
            # update asset class values
            data.loc[i, ['equity_value', 'bonds_value', 'debt_value']] = (
                data.loc[i-1, ['weight_equity', 'weight_bond', 'weight_debt']].values
                * data.loc[i-1, 'port_value']
                * data.loc[i, ['Monthly real stock rate', 'Monthly real gov bond rate', 'Monthly real margin rate']].values
            )
            # update portfolio value
            data.loc[i, 'port_value'] = data.loc[i, ['equity_value', 'bonds_value', 'debt_value']].sum()
            # compute portfolio returns
            data.loc[i, 'returns_port'] = data.loc[i, 'port_value'] / data.loc[i-1, 'port_value']
            # update cumulative portfolio returns
            data.loc[i, 'returns_port_cum'] = data.loc[i, 'returns_port'] * data.loc[i-1, 'returns_port_cum']
            # update the leverage
            data.loc[i, 'leverage'] = (
                (data.loc[i, ['equity_value', 'bonds_value']].sum())
                / data.loc[i, 'port_value']
            )
            # update asset class weights
            # this is where we could reset the leverage
            if row['date'] in rebalance_dates:
                data.loc[i, ['weight_equity', 'weight_bond', 'weight_debt']] = np.array([self.initial_leverage, 0.0, 1.0-self.initial_leverage])
            else:
                data.loc[i, ['weight_equity', 'weight_bond', 'weight_debt']] = (
                    data.loc[i, ['equity_value', 'bonds_value', 'debt_value']].values
                    / data.loc[i, 'port_value']
                )
            assert (data.loc[i, ['weight_equity', 'weight_bond', 'weight_debt']].sum() - 1.0) < 1e-8, f"""weights
            must sum to 1 but they sum to {data.loc[i, ['weight_equity', 'weight_bond', 'weight_debt']].sum()}"""
            
        data.loc[:, 'leverage_with_resets'] = (
            data.loc[:, ['weight_equity', 'weight_bond']].sum(axis=1)
            / data.loc[:, ['weight_equity', 'weight_bond', 'weight_debt']].sum(axis=1)
        )

        self.results=data
        self.measure_results()

    def measure_results(self):
        """measure performance of the backtest"""
        results = deepcopy(self.results)
        delta = relativedelta.relativedelta(results.iloc[-1, :].loc['date'], results.iloc[0, :].loc['date'])
        years = delta.years + delta.months/12

        # max drawdown
        i = np.argmax((results['port_value'].cummax() - results['port_value']).values) # end of the period
        j = np.argmax(results.iloc[:i, :].loc[:, 'port_value'].values) # start of period
        max_drawdown = (results.iloc[i, :].loc['port_value'] - results.iloc[j, :].loc['port_value']) / results.iloc[j, :].loc['port_value']

        stats = pd.DataFrame(
            {'leverage': self.initial_leverage,
             'equity_compound_return_per_annum': (1 + (results['Monthly real stock rate'].cumprod().values -1)[-1])**(1/years),
             'levered_equity_compound_return_per_annum': (1 + self.initial_leverage * (results['Monthly real stock rate'].cumprod().values -1)[-1])**(1/years),
             'arithmetic_return_per_annum': (results.loc[:, 'returns_port'] - 1).mean() * 12,
             'compound_return_per_annum': results.iloc[-1, :].loc['returns_port_cum']**(1/years) - 1,
             'volatility_per_annum': results.loc[:, 'returns_port'].std() * (12**0.5),
             'end_value_of_$1': results.iloc[-1, :].loc['port_value'],
             'max_drawdown': max_drawdown},
             index=[0]
        )
        self.stats = stats

    def get_trend_index_data(self):
        """Read in returns of Soc Gen Trend Index and Barclay's BTOP 50 Trend Index"""
        data_filename_socgen = "lc_investing/data/socgen_trend_index_returns.xlsx"
        trend_socgen = (
            pd.read_excel(data_filename_socgen)
            .drop(columns=['index level'])
            .rename(columns={'Date': 'date', 'SG Trend Index': 'returns'})
        )
        trend_socgen.insert(1, column='id', value='NEIXCTAT')

        data_filename_barclays = "lc_investing/data/BTOP50_Index_historical_data.xls"
        trend_barclays = pd.read_excel(data_filename_barclays, skiprows=1)
        month_map = dict(zip(trend_barclays.columns[1:],
                             ['-'+str(strptime(m, '%b').tm_mon)+'-01' for m in trend_barclays.columns[1:]]))
        trend_barclays = (
            trend_barclays
            .rename(columns={'Unnamed: 0': 'date'})
            .iloc[0:38, :]
            .melt(id_vars='date')
            .assign(date=lambda df: df['date'].astype(int).astype(str))
            .assign(date=lambda df: df['date'] + df['variable'].replace(month_map))
            .assign(date=lambda df: pd.to_datetime(df['date']))
            .drop(columns=['variable'])
            .rename(columns={'value': 'returns'})
            .assign(returns=lambda df: df['returns'].astype(float))
        )
        trend_barclays.insert(1, column='id', value='BTOP50')
        
        return trend_socgen, trend_barclays
    
    def get_index_data(self):
        """index data, stocks and bonds"""
        data = yf_retrieve_data(tickers=['IVV','INX','SPX','AGG','DBMF','KMLM','RSST'])
        return data
    
def yf_retrieve_data(tickers: List[str]):
  """Retrive price data from Yahoo Finance"""
  dataframes = []

  for ticker_name in tickers:
    ticker = yf.Ticker(ticker_name)
    history = ticker.history(period='10y')

    if (history.shape[0] > 0) and history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
      history = history.iloc[1:]
  
    assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
    # history['returns'] = history['Close'].pct_change()
    history_monthly = (
        history['Close']
        .resample('ME')
        # .resample('D')
        .ffill()
        .pct_change()
        .reset_index()
        .rename(columns={'Close': 'returns', 'Date': 'date'})
    )
    history_monthly.insert(1, value=ticker_name, column='id')
    dataframes.append(history_monthly)

  returns = pd.concat(dataframes, axis=0)
  return returns

def adjust_to_real(df: pd.DataFrame):
    """turn a returns series into real values"""
    # get CPI numbers
    # https://pieriantraining.com/exploring-inflation-data-with-python/
    # This sometimes take awhile!
    # cpi.update()
    # all items
    cpi_items_df = cpi.series.get(seasonally_adjusted=False).to_dataframe()
    # filter to only the monthly data we need
    cpi_items_df_2 = cpi_items_df[cpi_items_df['period_type']=='monthly']
    cpi_df = (
        cpi_items_df_2
        .assign(date=lambda df: pd.to_datetime(df['date']) - MonthEnd(1)) # assign to month end
        .loc[:, ['date', 'value']]
        .sort_values('date', ascending=True)
        .assign(inflation=lambda df: df['value'].pct_change())
        .rename(columns={'value': 'CPI'})
    )
    df1 = df.merge(cpi_df, on='date', how='left')
    # equation 1.15 here: https://bookdown.org/compfinezbook/introcompfinr/AssetReturnCalculations.html
    # simple returns (R)
    df1.loc[:, 'returns_real'] = ((1 + df1['returns']) / (1 + df1['inflation'])) - 1
    return df1.loc[:, ['date', 'id', 'returns_real']]

