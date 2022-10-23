import bisect
import numpy as np
import pandas as pd
from typing import List


def create_monthly_data(margin_call_info: pd.DataFrame,
                        data_folder: str,
                        bondadj: float,
                        margadj: float,
                        stockadj: float,
                        marginmonths: List,
                        marginreturn: List,
                        margincutoff: List,
                        cap: float):
    """Create monthly data"""
    
    base_file = data_folder + 'monthly_data.csv'

    df = pd.read_csv(base_file)

    # df.loc[:, 'Months_beginning_Jan_1871'] = pd.to_numeric(df.loc[:, 'Months_beginning_Jan_1871'])

    # return df

    df.loc[:, 'Monthly_CPI_lead1'] = df.loc[:, 'Monthly_CPI'].shift(-1)
    
    df.loc[:, 'Prospective_monthly_inflation_rate'] = df.loc[:, 'Monthly_CPI_lead1'] / df.loc[:, 'Monthly_CPI']
    df.loc[:, 'Annualized_adjusted_gov_bond_rate'] = (df.loc[:, 'Monthly_nom_gov_bond_rate']+1)**12 + bondadj
    df.loc[:, 'Annualized_adjusted_margin_rate'] = (df.loc[:, 'Monthly_nom_margin_rate']+1)**12 + margadj
    df.loc[:, 'Annualized_adjusted_stock_return'] = (df.loc[:, 'Monthly_nom_stock_return']+1)**12 + stockadj
    
    # =IF(A686>=685,LOOKUP(A686,marginmonths,marginreturn),1)
    def lookup(value: float, series1: np.array, series2):
      index = bisect.bisect_left(series1, value)
      if index == 0:
        if value < series1[0]:
          return None
        elif value == series1[0]:
          return series2[0]
        else:
          raise Exception("RH: this shouldn't happen")
      else:
        return series2[index - 1]

    df.loc[df.Months_beginning_Jan_1871 >= 685, 'Stock_Return_If_Margin_Call'] = [lookup(m, marginmonths, marginreturn) for m in df.loc[df.Months_beginning_Jan_1871 >= 685, 'Months_beginning_Jan_1871']]
    df.loc[df.Months_beginning_Jan_1871 < 685, 'Stock_Return_If_Margin_Call'] = 1
    
    df.loc[:, 'Monthly_real_gov_bond_rate'] = df.loc[:, 'Annualized_adjusted_gov_bond_rate']**(1/12) * df['Monthly_CPI']/df['Monthly_CPI_lead1'] - 1
    df.loc[:, 'Monthly_real_margin_rate'] = df.loc[:, 'Annualized_adjusted_margin_rate']**(1/12) * df['Monthly_CPI']/df['Monthly_CPI_lead1'] - 1
    df.loc[:, 'Monthly_real_stock_rate'] = df.loc[:, 'Annualized_adjusted_stock_return']**(1/12) * df['Monthly_CPI']/df['Monthly_CPI_lead1'] - 1
  
    df.loc[:, 'Log_Annualized_adjusted_stock_return'] = np.log(df.loc[:, 'Annualized_adjusted_stock_return'])

    df.loc[df.Months_beginning_Jan_1871 >= 685, 'Margin_Call_Cutoff'] = [lookup(m, marginmonths, margincutoff) for m in df.loc[df.Months_beginning_Jan_1871 >= 685, 'Months_beginning_Jan_1871']]
    df.loc[df.Months_beginning_Jan_1871 < 685, 'Margin_Call_Cutoff'] = cap + 1
    
    df.loc[:, 'Margin_Call_Real_Stock_Return'] = df.loc[:, 'Stock_Return_If_Margin_Call']**(1/12) * df['Monthly_CPI']/df['Monthly_CPI_lead1'] - 1
    df.loc[:, 'One_plus_monthly_real_stock_rate'] = df.loc[:, 'Monthly_real_stock_rate'] + 1
    df.loc[:, 'One_plus_monthly_real_gov_bond_rate'] = df.loc[:, 'Monthly_real_gov_bond_rate'] + 1
    
    return df