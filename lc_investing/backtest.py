from copy import deepcopy
import datetime
from dateutil import relativedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import os
import sys
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
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
        data = pd.read_excel(
            data_filename,
            sheet_name='Monthly data',
            header=0)
        # assign dates using last day of the month
        data['Monthly years'] = pd.to_datetime(data.loc[:, 'Monthly years'].astype(str), format="%Y.%m") + MonthEnd(0)
        data = (
            data
            .rename(columns={'Monthly years': 'date'})
            .iloc[0:1662, :]  # remove last data points since they aren't accurate
        )
        # filter to desired start date
        data1 = (
            data
            .loc[:, ['date', 'Monthly real stock rate', 'Monthly real gov bond rate', 'Monthly real margin rate']]
            .query(f"date >= '{start_date}'")
            .query(f"date <= '{end_date}'")
            .reset_index(drop=True)
        )
        # change simple returns R to 1+R
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

