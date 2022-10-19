import numpy as np
import pandas as pd

from lc_investing.contributions import create_contributions, create_income_contributions
from lc_investing.create_cohorts import create_cohorts
from lc_investing.monthly_data import create_monthly_data
from lc_investing.pe_multiplier import caclulate_pe_10, caclulate_pe_10_multiplier
from lc_investing.margin_call_info import create_margin_call_info
from lc_investing.utils import initialize_cohort_table, create_data_month


class Simulation:

  def __init__(self,
               data_folder='/content/drive/Othercomputers/My MacBook Air/Taxes_and_other_forms/lifecycle_investing/lc_investing/data/',
               cap=2,
               incomemult=2.35217277377134,
               contrate=0.04,
               ssreplace=0.0,
               rmm=0.00213711852838,
               rfm=0.00211039468707308,
               birthint=0.9,
               birthfin=0.5):

    self.data_folder=data_folder
    self.cap=cap
    self.incomemult=incomemult
    self.contrate=contrate
    self.ssreplace=ssreplace
    self.rmm=rmm
    self.rfm=rfm
    self.birthint=birthint
    self.birthfin=birthfin
    
    self.cohorts = create_cohorts()
    
    self.margin_call_info = create_margin_call_info(data_folder=self.data_folder)

    self.monthly_data = create_monthly_data(data_folder=self.data_folder,
                                            margin_call_info=self.margin_call_info,
                                            bondadj=0,
                                            margadj=0,
                                            stockadj=0,
                                            marginmonths=self.margin_call_info['Month'].tolist(),
                                            marginreturn=self.margin_call_info['Adjusted_Annual_Worst_This_Month'].tolist(),
                                            margincutoff=self.margin_call_info['Margin_Call_Cutoff_Point'].tolist(),
                                            cap=self.cap)
    
    self.percentage_target = self.calc_percentage_target(cohorts=self.cohorts, 
                                                         birthint=self.birthint,
                                                         birthfin=self.birthfin)

    self.data_month = create_data_month(self.cohorts)

    self.real_return = self.calc_real_return(monthly_data=self.monthly_data,
                                             percentage_target=self.percentage_target,
                                             data_month=self.data_month)

    self.income_contrib = create_income_contributions(data_folder=self.data_folder,
                                                      incomemult=self.incomemult,
                                                      contrate=self.contrate)

    self.contributions = create_contributions(income_contrib=self.income_contrib,
                                              ssreplace=self.ssreplace,
                                              rmm=self.rmm,
                                              rfm=self.rfm)

    self.retirement_savings_before_period = self.calc_retirement_savings_before_period(cohorts=self.cohorts,
                                                                                       contributions=self.contributions,
                                                                                       real_return=self.real_return)

  def calc_retirement_savings_before_period(self,
                                            cohorts: pd.DataFrame, 
                                            contributions: pd.DataFrame,
                                            real_return: pd.DataFrame):
    """Calculate the retirement savings before the period"""

    cohorts_2 = pd.DataFrame(data=np.empty((cohorts.shape[0], 528)),
                             columns=list(range(1,529)))

    cohorts_3 = pd.concat([cohorts, cohorts_2], axis=1)

    cohorts_melted = pd.melt(cohorts_3, 
                             id_vars=['cohort_num', 'begins_work', 'retire'],
                             var_name='period_num',
                             value_name='placeholder')

    df1 = pd.merge(cohorts_melted,
                   contributions.loc[:, ['Months', 'Monthly_Contribution']],
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
        (1 + real_return.loc[:, c-1])  # multiply by returns

    sstotal = contributions.Monthly_Contribution.values[-1]

    df1_wide.loc[:, 'Final'] = df1_wide.loc[:, 528] * (1 + real_return.loc[:, 528]) + sstotal
    
    return df1_wide
    

  def calc_percentage_target(self,
                             cohorts: pd.DataFrame, 
                             birthint: float = 0.9,
                             birthfin: float = 0.5):

    cohorts_2 = pd.DataFrame(data=np.zeros((cohorts.shape[0], 528)),
                             columns=list(range(1,529)))

    cohorts_3 = pd.concat([cohorts, cohorts_2], axis=1)

    cohorts_melted = pd.melt(cohorts_3, 
                             id_vars=['cohort_num', 'begins_work', 'retire'],
                             var_name='period_num',
                             value_name='placeholder')

    cohorts_melted.loc[:, 'percentage_target'] = birthint - (cohorts_melted['period_num'] - 1) * (birthint - birthfin)/528

    percentage_target = pd.pivot(cohorts_melted,
                                 index=['cohort_num', 'begins_work', 'retire'],
                                 columns=['period_num'],
                                 values=['percentage_target']) #.reset_index(drop=True)

    percentage_target.columns = percentage_target.columns.droplevel(0)
    percentage_target.columns.name = None

    percentage_target = percentage_target.reset_index()

    return percentage_target


  def calc_real_return(self,
                       monthly_data: pd.DataFrame,
                       percentage_target: pd.DataFrame,
                       data_month: pd.DataFrame):

    data_month_melted = pd.melt(data_month, 
                                id_vars=['cohort_num', 'begins_work', 'retire'],
                                var_name='period_num',
                                value_name='month')

    df3 = pd.merge(data_month_melted, 
                   monthly_data.loc[:, ['Months_beginning_Jan_1871', 
                                        'Monthly_real_gov_bond_rate',	
                                        'Monthly_real_margin_rate',	
                                        'Monthly_real_stock_rate']],
                   left_on='month',
                   right_on='Months_beginning_Jan_1871'). \
            sort_values(['cohort_num', 'period_num', 'begins_work'])

    percentage_target_melted = pd.melt(percentage_target, 
                                       id_vars=['cohort_num', 'begins_work', 'retire'],
                                       var_name='period_num',
                                       value_name='percentage_target')

    df4 = pd.merge(df3, 
                  percentage_target_melted.loc[:, ['cohort_num', 'period_num', 'percentage_target']],
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

    df5.columns.name = None

    df5 = df5.reset_index()

    return df5




def Amount_in_stock(cohort: pd.DataFrame, 
                    retirement_savings_before_period: pd.DataFrame,
                    percentage_target: pd.DataFrame):
    pass
def Present_value_of_accumulation(cohort: pd.DataFrame):
    pass
def Present_value_of_accumulation(cohort: pd.DataFrame, 
                                  retirement_savings_before_period: pd.DataFrame,
                                  rfm: float):
    pass
def Utility(cohort: pd.DataFrame, 
            retirement_savings_before_period: pd.DataFrame,
            crracons):
    pass
def Herfindal_Hirshman_Index_Calculation(cohort: pd.DataFrame, 
                                         amount_in_stock: pd.DataFrame):
    pass
def Payment_Stream(cohort: pd.DataFrame, 
                   contributions: pd.DataFrame):
    pass
