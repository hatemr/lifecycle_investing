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
               startper=1,
               lambda1=0.83030407,
               lambda2=0.83030407,
               cap=2,
               requirement=0,
               incomemult=2.35217277377134,
               contrate=0.04,
               ssreplace=0.0,
               rmm=0.00213711852838,
               rfm=0.00211039468707308,
               PEadjust=0,
               maxsam=2,
               minsam=0,
               inheritance_indicator=0,
               inheritance_amount=5000,
               lambdaearly=2,
               lambdacons=0.75):

    self.data_folder=data_folder
    self.startper=startper
    self.lambda1=lambda1
    self.lambda2=lambda2
    self.cap=cap
    self.requirement=requirement
    self.incomemult=incomemult
    self.contrate=contrate
    self.ssreplace=ssreplace
    self.rmm=rmm
    self.rfm=rfm
    self.PEadjust=PEadjust
    self.maxsam=maxsam
    self.minsam=minsam
    self.inheritance_indicator=inheritance_indicator
    self.inheritance_amount=inheritance_amount
    self.lambdaearly=lambdaearly
    self.lambdacons=lambdacons
                                                      

    self.cohorts = create_cohorts()
    
    # self.retirement_savings_before_period = initialize_cohort_table(self.cohorts)
    # self.percentage_target = initialize_cohort_table(self.cohorts)
    # self.real_returns = initialize_cohort_table(self.cohorts)
    # self.real_return_for_ages_0_22 = initialize_cohort_table_birth()
    # self.inheritance_from_ages_0_22 = initialize_cohort_table_birth()
    # self.inheritance_from_ages_0_22.loc[self.inheritance_from_ages_0_22.cohort_num >= 24, 0] = self.inheritance_amount if self.inheritance_indicator==1 else 0

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
    
    # # from 'PE Multiplier' sheet
    # self.pe_10 = caclulate_pe_10(data_folder=self.data_folder)

    # # from 'PE Multiplier' sheet
    # self.pe_10_samuelson = caclulate_pe_10_multiplier(data_folder=self.data_folder,
    #                                                   maxsam=self.maxsam,
    #                                                   minsam=self.minsam,
    #                                                   PEadjust=self.PEadjust)

    # # from 'Lifecycle strategy' sheet
    # self.df_pe_depending_on_period = PE_10_depending_on_period(self.data_month, 
    #                                                            self.pe_10)
    
    # # from 'Lifecycle strategy' sheet
    # self.pe_multiplier = PE_Multiplier(df_PE_10_depending_on_period=self.df_pe_depending_on_period,
    #                                    pe_10_samuelson=self.pe_10_samuelson)

    # self.income_contrib = create_income_contributions(data_folder=self.data_folder,
    #                                                      incomemult=self.incomemult,
    #                                                      contrate=self.contrate)

    # self.contributions = create_contributions(income_contrib=self.income_contrib,
    #                                           ssreplace=self.ssreplace,
    #                                           rmm=self.rmm,
    #                                           rfm=self.rfm)

    # # static table
    # self.Samuelson_Share_for_ages_0_22 = Samuelson_Share_for_ages_0_22(lambdaearly=self.lambdaearly)

    self.percentage_target = self.calc_percentage_target(cohorts=self.cohorts, 
                                                         lambdacons=self.lambdacons)

    self.data_month = create_data_month(self.cohorts)

    # self.real_return = self.calc_real_return(monthly_data=self.monthly_data,
    #                                          percentage_target=self.percentage_target,
    #                                          data_month=self.data_month)

    # sefl.income_contrib = create_income_contributions(incomemult=self.incomemult,
    #                                                   contrate=self.contrate)

    # self.contributions = create_contributions(income_contrib=self.income_contrib,
    #                                           ssreplace=self.ssreplace,
    #                                           rmm=self.rmm,
    #                                           rfm=self.rfm)

    # self.retirement_savings_before_period = self.calc_retirement_savings_before_period(df_cohorts=self.cohorts,
    #                                                                                       contributions=self.contributions,
    #                                                                                       real_return=self.real_return)


  def calc_retirement_savings_before_period(self,
                                            cohorts: pd.DataFrame, 
                                            contributions: pd.DataFrame,
                                            real_return: pd.DataFrame):
    """"Calculates the retirement savings before the period"""

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

    for c in range(2, 529):
      # last period, plus contributions, times returns
      df1_wide.loc[:, c] = (df1_wide.loc[:, c-1] + df1_wide.loc[:, c]) * \
        (1 + real_return.loc[:, c-1])  # multiply by returns

    sstotal = contributions.Monthly_Contribution.values[-1]

    df1_wide.loc[:, 'Final'] = df1_wide.loc[:, 528] * (1 + real_return.loc[:, 528]) + sstotal
    
    return df1_wide
  

  def calc_percentage_target(self,
                             cohorts: pd.DataFrame, 
                             lambdacons: float):
    df1 = cohorts.copy()

    return df1
    perc_targ = pd.DataFrame(data=lambdacons * np.ones((df1.shape[0],528)),
                             columns=list(range(1,529)))

    df2 = pd.concat([cohorts, perc_targ], axis=1)

    return df2


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

    return df3

    percentage_target_melted = pd.melt(percentage_target, 
                                       id_vars=['cohort_num', 'begins_work', 'retire'],
                                       var_name='period_num',
                                       value_name='percentage_target')

    df4 = pd.merge(df3, 
                  percentage_target_melted.loc[:, ['cohort_num', 'period_num', 'percentage_target']],
                  left_on=['cohort_num','period_num'],
                  right_on=['cohort_num','period_num'])

    return df4

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

    return df5





def Amount_in_stock(df_cohort: pd.DataFrame, 
                    df_retirement_savings_before_period: pd.DataFrame,
                    percentage_target: pd.DataFrame):
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
                   contributions: pd.DataFrame):
    pass
