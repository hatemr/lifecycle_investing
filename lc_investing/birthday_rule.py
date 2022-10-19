import numpy as np
import pandas as pd

def Retirement_savings_before_period(df_cohorts: pd.DataFrame, 
                                     df_contributions: pd.DataFrame,
                                     df_real_return: pd.DataFrame):

  df_cohorts_2 = pd.DataFrame(data=np.empty((df_cohorts.shape[0], 528)),
                              columns=list(range(1,529)))

  df_cohorts_3 = pd.concat([df_cohorts, df_cohorts_2], axis=1)

  df_cohorts_melted = pd.melt(df_cohorts_3, 
                              id_vars=['cohort_num', 'begins_work', 'retire'],
                              var_name='period_num',
                              value_name='placeholder')

  df1 = pd.merge(df_cohorts_melted,
                 df_contributions.loc[:, ['Months', 'Monthly_Contribution']],
                 left_on='period_num',
                 right_on='Months'). \
    drop(columns=['placeholder','Months'])

  df1_wide = df1.pivot(index=['cohort_num', 'begins_work', 'retire'], 
                       columns='period_num', 
                       values='Monthly_Contribution')

  df1_wide.columns.name = None
  df1_wide = df1_wide.reset_index()

  for c in range(2, 529):
    # last period, plus contributions, times returns
    df1_wide.loc[:, c] = (df1_wide.loc[:, c-1] + df1_wide.loc[:, c]) * \
      (1 + df_real_return.loc[:, c-1])  # multiply by returns

  sstotal = df_contributions.Monthly_Contribution.values[-1]

  df1_wide.loc[:, 'Final'] = df1_wide.loc[:, 528] * (1 + df_real_return.loc[:, 528]) + sstotal
  
  return df1_wide
  

def Percentage_Target(df_cohorts: pd.DataFrame, 
                      birthint: float = 0.9,
                      birthfin: float = 0.5):

  df_cohorts_2 = pd.DataFrame(data=np.zeros((df_cohorts.shape[0], 528)),
                              columns=list(range(1,529)))

  df_cohorts_3 = pd.concat([df_cohorts, df_cohorts_2], axis=1)

  df_cohorts_melted = pd.melt(df_cohorts_3, 
                              id_vars=['cohort_num', 'begins_work', 'retire'],
                              var_name='period_num',
                              value_name='placeholder')

  df_cohorts_melted.loc[:, 'percentage_target'] = birthint - (df_cohorts_melted['period_num'] - 1) * (birthint - birthfin)/528

  df_percentage_target = pd.pivot(df_cohorts_melted,
                                  index=['cohort_num', 'begins_work', 'retire'],
                                  columns=['period_num'],
                                  values=['percentage_target']) #.reset_index(drop=True)

  df_percentage_target.columns = df_percentage_target.columns.droplevel(0)
  df_percentage_target.columns.name = None

  df_percentage_target = df_percentage_target.reset_index()

  return df_percentage_target


def Real_Return(df_cohorts: pd.DataFrame, 
                df_monthly_data: pd.DataFrame,
                df_percentage_target: pd.DataFrame,
                df_data_month: pd.DataFrame):

  df_data_month_melted = pd.melt(df_data_month, 
                                 id_vars=['cohort_num', 'begins_work', 'retire'],
                                 var_name='period_num',
                                 value_name='month')

  df3 = pd.merge(df_data_month_melted, 
                 df_monthly_data.loc[:, ['Months_beginning_Jan_1871', 
                                         'Monthly_real_gov_bond_rate',	
                                         'Monthly_real_margin_rate',	
                                         'Monthly_real_stock_rate']],
                 left_on='month',
                 right_on='Months_beginning_Jan_1871'). \
          sort_values(['cohort_num', 'period_num', 'begins_work'])

  df_percentage_target_melted = pd.melt(df_percentage_target, 
                                        id_vars=['cohort_num', 'begins_work', 'retire'],
                                        var_name='period_num',
                                        value_name='percentage_target')

  df4 = pd.merge(df3, 
                 df_percentage_target_melted.loc[:, ['cohort_num', 'period_num', 'percentage_target']],
                 left_on=['cohort_num','period_num'],
                 right_on=['cohort_num','period_num'])

  # if allocation is greater than 100%, use margin rate
  df4.loc[df4.percentage_target > 1, 'monthly_real_return'] = df4.loc[df4.percentage_target > 1, 'Monthly_real_margin_rate']
  df4.loc[df4.percentage_target <= 1, 'monthly_real_return'] = df4.loc[df4.percentage_target <= 1, 'Monthly_real_gov_bond_rate']

  df4.loc[:, 'monthly_real_return'] = (1 - df4.loc[:, 'percentage_target']) * df4.loc[:, 'monthly_real_return']
  df4.loc[:, 'monthly_real_return'] = df4.loc[:, 'monthly_real_return'] + df4.loc[:, 'percentage_target'] * df4.loc[:, 'Monthly_real_stock_rate']
  df4.loc[:, 'comparison_dummy'] = -1
  df4.loc[:, 'monthly_real_return'] = df4.loc[:, ['comparison_dummy','monthly_real_return']].max(axis=1)

  df5 = df4.pivot(index = ['cohort_num',	'begins_work',	'retire'], 
                  columns = ['period_num'], 
                  values = 'monthly_real_return')

  # df5.columns = df5.columns.droplevel(0)
  df5.columns.name = None

  df5 = df5.reset_index()

  return df5


def Data_month(df_cohort: pd.DataFrame):
    """Creates indexes for months"""
    df1 = df_cohort.copy()

    new_cols = [c + 12*(df1.cohort_num - 1) for c in range(1,529)]
    
    df2 = pd.concat([df1,
                     pd.DataFrame(np.array(new_cols).T,
                                  columns=list(range(1, len(new_cols)+1)))], axis=1)
    
    return df2


def Amount_in_stock(df_cohort: pd.DataFrame, 
                    df_retirement_savings_before_period: pd.DataFrame,
                    df_percentage_target: pd.DataFrame):
    pass
def Present_value_of_accumulation(df_cohort: pd.DataFrame, 
                                     ):
    pass
def Present_value_of_accumulation(df_cohort: pd.DataFrame, 
                                  df_retirement_savings_before_period: pd.DataFrame,
                                  rfm: float):
    pass
def Utility(df_cohort: pd.DataFrame, 
            df_retirement_savings_before_period: pd.DataFrame,
            crracons):
    pass
def Herfindal_Hirshman_Index_Calculation(df_cohort: pd.DataFrame, 
                                         df_amount_in_stock: pd.DataFrame):
    pass
def Payment_Stream(df_cohort: pd.DataFrame, 
                   df_contributions: pd.DataFrame):
    pass
