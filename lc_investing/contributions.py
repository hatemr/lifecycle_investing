import numpy as np
import pandas as pd

from .utils import npv


def create_income_contributions(data_folder: str,
                                incomemult: float,
                                contrate: float):
    """
    
    incomemult: multiplier use to get your actual annual income
    contrate: contribution rate. The fraction of your gross annual 
        income that you invest
    """
    base_file = data_folder + 'contributions.csv'

    df = pd.read_csv(base_file)

    df.loc[:, 'Income_Stream'] = df.loc[:, 'Social_Security_Wage_Profile'] * incomemult

    df.loc[:, 'Yearly_income_contribution'] = df.loc[:, 'Income_Stream'] * contrate

    return df

def create_contributions(income_contrib: pd.DataFrame,
                         ssreplace: float,
                         rmm: float,
                         rfm: float):
    """"""

    df = pd.DataFrame({'Months': list(range(1,529))})

    df.loc[:, 'Age'] = np.array([i // 12 for i in range(0,528)]) + 23

    df1 = pd.merge(df, 
                   income_contrib.loc[:, ['Age', 'Yearly_income_contribution']].copy(), 
                   on='Age', 
                   how='left')

    df1.loc[:, 'Monthly_Contribution'] = df1.loc[:, 'Yearly_income_contribution']/12
    
    df2 = df1.tail(1).copy()
    df2.loc[:,:] = np.nan
    df2.loc[:, 'Monthly_Contribution'] = 100000 * ssreplace * 18.8

    df1 = df1.append(df2, ignore_index=False)
    
    cashflows = df1.Monthly_Contribution.values.tolist()

    NPVs = np.array([npv(rmm, [0] + cashflows[i:]) for i in range(len(cashflows))]) * (1 + rmm)
    df1.loc[:, 'PV_of_remain_contrib_at_margin_rate'] = NPVs
    
    NPVs_riskfree = NPVs = np.array([npv(rfm, [0] + cashflows[i:]) for i in range(len(cashflows))]) * (1 + rfm)
    df1.loc[:, 'PV_of_remain_contrib_at_risk_free_rate'] = NPVs_riskfree

    df1.loc[:, 'Yearly_Contribution'] = df1.loc[:, 'Monthly_Contribution'] * 12
  
    return df1