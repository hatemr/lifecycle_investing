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
              #  data_folder='/content/drive/Othercomputers/My MacBook Air/Taxes_and_other_forms/lifecycle_investing/lc_investing/data/',
               data_folder='./lc_investing/data/',
               startper=1,
               lambda1=0.83030407,  # initial Samuelson share
               lambda2=0.83030407,  # second Samuelson share
               cap=2,				# leverage cap
               requirement=0,		# margin requirement
               incomemult=2.35217277377134,  # multiplies Social Security based income to achieve a realistic income
               contrate=0.04,		# contribution rate. Fraction of monthly income saved and invested
               ssreplace=0.0,		# Social Security replacement rate (?)
               rmm=0.00213711852838, 
               rfm=0.00211039468707308, 
               PEadjust=0,  		# 0=unadjusted, 1=adjusted
               maxsam=2,			# maximum Samuelson share
               minsam=0,			# minimum Samuelson share
               inheritance_indicator=0,  # inheritance/early start indicator. 1=inheritance given at birth, 0=no inheritance
               inheritance_amount=5000,  # inheritance amount is $500, $1000, $5000
               lambdaearly=2):

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
                                                      

    self.cohorts = create_cohorts()

    self.retirement_savings_before_period = initialize_cohort_table(self.cohorts)
    self.percentage_target = initialize_cohort_table(self.cohorts)
    self.real_returns = initialize_cohort_table(self.cohorts)
    self.real_return_for_ages_0_22 = initialize_cohort_table_birth()
    self.inheritance_from_ages_0_22 = initialize_cohort_table_birth()
    self.inheritance_from_ages_0_22.loc[self.inheritance_from_ages_0_22.cohort_num >= 24, 0] = self.inheritance_amount if self.inheritance_indicator==1 else 0

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
    
    # from 'PE Multiplier' sheet
    self.pe_10 = caclulate_pe_10(data_folder=self.data_folder)
    
    # from 'PE Multiplier' sheet
    self.pe_10_samuelson = caclulate_pe_10_multiplier(data_folder=self.data_folder,
                                                      maxsam=self.maxsam,
                                                      minsam=self.minsam,
                                                      PEadjust=self.PEadjust)

    self.data_month = create_data_month(self.cohorts)
    
    # from 'Lifecycle strategy' sheet
    self.pe_depending_on_period = PE_10_depending_on_period(self.data_month, 
                                                            self.pe_10)
    
    # from 'Lifecycle strategy' sheet
    self.pe_multiplier = PE_Multiplier(PE_10_depending_on_period=self.pe_depending_on_period,
                                       pe_10_samuelson=self.pe_10_samuelson)

    self.income_contrib = create_income_contributions(data_folder=self.data_folder,
                                                      incomemult=self.incomemult,
                                                      contrate=self.contrate)

    self.contributions = create_contributions(income_contrib=self.income_contrib,
                                              ssreplace=self.ssreplace,
                                              rmm=self.rmm,
                                              rfm=self.rfm)

    # static table
    self.Samuelson_Share_for_ages_0_22 = Samuelson_Share_for_ages_0_22(lambdaearly=self.lambdaearly)
    
    

  def calc_retirement_savings_before_period(self,
                                            period: int = 528):

    """
    Calculate the retirement savings before period. Then compute the 
    percentage_target and real_returns for the period.
    """

    assert period >= 1, 'period must be >=1'
    
    for per in range(1, period+1):
      
      # must run in this order
      self.retirement_savings_before_period.loc[:, per] = (self.contributions.loc[self.contributions.Months==per, 'Monthly_Contribution'].values[0] + self.retirement_savings_before_period.loc[:, per-1]) * (1 + self.real_returns.loc[:, per-1])
      
      self.calc_percentage_target(period=per)
      
      self.calc_real_return(period=per, requirement=self.requirement)

      # right placement?
      self.calc_real_return_for_ages_0_22(period=per)

      # must be after calc_real_return_for_ages_0_22
      self.calc_inheritance_from_ages_0_22(period=per)

    self.sstotal = self.contributions.Monthly_Contribution.values[-1]
    self.retirement_savings_before_period.loc[:, 'FINAL'] = self.retirement_savings_before_period.loc[:, period] * (1 + self.real_returns.loc[:, period]) + self.sstotal

    return


  def calc_percentage_target(self, period=1):

    """
    Calculate the percentage target for period. Assumes that 
    retirement_savings_before_period for period has already been computed.
    """

    assert period >= 1, 'period must be >=1'

    if period >= self.startper:

      numerator1 = (self.lambda1 * self.pe_multiplier.loc[:, period]) * (self.contributions.loc[self.contributions.Months==period, 'PV_of_remaining_contributions_at_the_margin_rate'].values[0] + self.retirement_savings_before_period.loc[:, period])
      denominator1 = (self.retirement_savings_before_period.loc[:, period] + self.contributions.loc[self.contributions.Months==period, 'Monthly_Contribution'].values[0])

      df1 = pd.DataFrame(data={'retirement_savings_before_period': self.retirement_savings_before_period.loc[:, period],  # DU4
                               'PV_of_remaining_contributions_at_the_margin_rate': self.contributions.loc[self.contributions.Months==period, 'PV_of_remaining_contributions_at_the_margin_rate'].values[0],  # INDEX(Contributions!$H$2:$H$529,DC$101)
                               'pe_multiplier': self.pe_multiplier.loc[:, period],  # DU1192
                               'numerator1': numerator1,
                               'denominator1': denominator1,
                               'comparison1': numerator1/denominator1})
      
	  # leverage cap (e.g. cap=2)
      df1.loc[:, 'cap'] = self.cap

      numerator2 = (self.lambda2 * self.pe_multiplier.loc[:, period]) * (self.contributions.loc[self.contributions.Months==period, 'PV_of_remaining_contributions_at_the_risk_free_rate'].values[0] + self.retirement_savings_before_period.loc[:, period])
      denominator2 = (self.retirement_savings_before_period.loc[:, period] + self.contributions.loc[self.contributions.Months==period, 'Monthly_Contribution'].values[0])
      
      df1.loc[:, 'numerator2'] = numerator2
      df1.loc[:, 'denominator2'] = denominator2

      df1.loc[:, 'comparison2'] = numerator2/denominator2
      df1.loc[:, 'comparision3'] = 1

      df1.loc[df1.comparison1 > 1, 'perc_targ'] = df1.loc[:, ['cap','comparison1']].min(axis=1)
      df1.loc[df1['comparison1'] <= 1, 'perc_targ'] = df1.loc[:, ['comparision3','comparison2']].min(axis=1)

      self.percentage_target.loc[:, period] = df1.loc[:, 'perc_targ']

    else:
      raise Exception('RH: todo')

    return


  def calc_real_return(self, 
                        period,
                        requirement):
    """
    Calculates the real return for period. It assumes that percentage_target
    for period has already been computed.
    """

    assert period >= 1, 'period must be >=1'

    data_month_melted = pd.melt(self.data_month.loc[:, ['cohort_num', 'begins_work', 'retire', period]], 
                                id_vars=['cohort_num', 'begins_work', 'retire'],
                                var_name='period_num',
                                value_name='month')

    df1 = pd.merge(data_month_melted, 
                   self.monthly_data.loc[:, ['Months_beginning_Jan_1871', 
                                             'Monthly_real_gov_bond_rate', # L	
                                             'Monthly_real_margin_rate', # M
                                             'Monthly_real_stock_rate', # N
                                             'Margin_Call_Cutoff', # P
                                             'Margin_Call_Real_Stock_Return']], # Q
                   left_on='month',
                   right_on='Months_beginning_Jan_1871'). \
      sort_values(['cohort_num', 'period_num', 'begins_work'])

    df1.loc[:, 'percentage_target'] = self.percentage_target.loc[:, period]

    option1 = (1 - df1.loc[:, 'percentage_target']) * df1.loc[:, 'Monthly_real_margin_rate'] - 1 + (1 + df1.loc[:, 'Margin_Call_Real_Stock_Return']) * ((1 - df1.loc[:, 'percentage_target'] * requirement) / (df1.loc[:, 'percentage_target'] - df1.loc[:, 'percentage_target'] * requirement))
    
    option2_sub = df1.loc[:, ['percentage_target', 'Monthly_real_margin_rate', 'Monthly_real_gov_bond_rate']]
    option2_sub.loc[option2_sub.percentage_target > 1, 'var1'] = option2_sub.loc[option2_sub.percentage_target > 1, 'Monthly_real_margin_rate']
    option2_sub.loc[option2_sub.percentage_target <= 1, 'var1'] = option2_sub.loc[option2_sub.percentage_target <= 1, 'Monthly_real_gov_bond_rate']
    
    df1.loc[:, 'option2_sub_var1'] = option2_sub.var1

    option2 = df1.percentage_target * df1.Monthly_real_stock_rate + (1 - df1.percentage_target) * (option2_sub.var1)
    
    df1.loc[:, 'option1'] = option1
    df1.loc[:, 'option2'] = option2

    df1.loc[df1.percentage_target > df1.Margin_Call_Cutoff, 'var2'] = df1.loc[df1.percentage_target > df1.Margin_Call_Cutoff, 'option1']
    df1.loc[df1.percentage_target <= df1.Margin_Call_Cutoff, 'var2'] = df1.loc[df1.percentage_target <= df1.Margin_Call_Cutoff, 'option2']

    df1.loc[:, 'base'] = -1
    df1.loc[:, 'real_returns'] = df1.loc[:, ['base', 'var2']].max(axis=1)

    self.real_returns.loc[:, period] = df1.loc[:, 'real_returns']

    return df1


  def calc_real_return_for_ages_0_22(self, period: int):
    """
    From 'Lifecycle strategy' sheet

    =MAX(-1, IF(F1515>INDEX('Monthly data'!$P$2:$P$1669,F301),(1-F1515)*INDEX('Monthly data'!$M$2:$M$1669,F301)-1+(1+INDEX('Monthly data'!$Q$2:$Q$1669,F301))*(1-F1515*Requirement)/(F1515-F1515*Requirement),F1515*INDEX('Monthly data'!$N$2:$N$1669,F301)+(1-F1515)*IF(F1515>1,INDEX('Monthly data'!$M$2:$M$1669,F301),INDEX('Monthly data'!$L$2:$L$1669,F301))))
    """
    
    assert period >= 1, 'period must be >=1'

    data_month_melted = pd.melt(self.data_month.loc[:, ['cohort_num', 'begins_work', 'retire', period]], 
                                id_vars=['cohort_num', 'begins_work', 'retire'],
                                var_name='period_num',
                                value_name='month')

    data_month_melted.loc[:, 'month'] = data_month_melted.loc[:, 'month'].shift(periods=23)

    df1 = pd.merge(data_month_melted, 
                   self.monthly_data.loc[:, ['Months_beginning_Jan_1871', 
                                             'Monthly_real_gov_bond_rate', # L	
                                             'Monthly_real_margin_rate', # M
                                             'Monthly_real_stock_rate', # N
                                             'Margin_Call_Cutoff', # P
                                             'Margin_Call_Real_Stock_Return']], # Q
                   left_on='month',
                   right_on='Months_beginning_Jan_1871'). \
                   sort_values(['cohort_num', 'period_num', 'begins_work'])

    # df1 = df1.loc[df1.cohort_num >= 24, :]

    df1.loc[:, 'samuelson_Share_for_ages_0_22'] = self.Samuelson_Share_for_ages_0_22.loc[:, period]
    
    option1 = (1 - df1.loc[:, 'samuelson_Share_for_ages_0_22']) * df1.loc[:, 'Monthly_real_margin_rate'] - 1 + (1 + df1.loc[:, 'Margin_Call_Real_Stock_Return']) * ((1 - df1.loc[:, 'samuelson_Share_for_ages_0_22'] * self.requirement) / (df1.loc[:, 'samuelson_Share_for_ages_0_22'] - df1.loc[:, 'samuelson_Share_for_ages_0_22'] * self.requirement))
    
    option2_sub = df1.loc[:, ['samuelson_Share_for_ages_0_22', 'Monthly_real_margin_rate', 'Monthly_real_gov_bond_rate']]
    option2_sub.loc[option2_sub.samuelson_Share_for_ages_0_22 > 1, 'var1'] = option2_sub.loc[option2_sub.samuelson_Share_for_ages_0_22 > 1, 'Monthly_real_margin_rate']
    option2_sub.loc[option2_sub.samuelson_Share_for_ages_0_22 <= 1, 'var1'] = option2_sub.loc[option2_sub.samuelson_Share_for_ages_0_22 <= 1, 'Monthly_real_gov_bond_rate']
    
    df1.loc[:, 'option2_sub_var1'] = option2_sub.var1

    option2 = df1.samuelson_Share_for_ages_0_22 * df1.Monthly_real_stock_rate + (1 - df1.samuelson_Share_for_ages_0_22) * (option2_sub.var1)
    
    df1.loc[:, 'option1'] = option1
    df1.loc[:, 'option2'] = option2

    df1.loc[df1.samuelson_Share_for_ages_0_22 > df1.Margin_Call_Cutoff, 'var2'] = df1.loc[df1.samuelson_Share_for_ages_0_22 > df1.Margin_Call_Cutoff, 'option1']
    df1.loc[df1.samuelson_Share_for_ages_0_22 <= df1.Margin_Call_Cutoff, 'var2'] = df1.loc[df1.samuelson_Share_for_ages_0_22 <= df1.Margin_Call_Cutoff, 'option2']

    df1.loc[:, 'base'] = -1
    df1.loc[:, 'real_return_for_ages_0_22'] = df1.loc[:, ['base', 'var2']].max(axis=1)

    self.real_return_for_ages_0_22.loc[self.real_return_for_ages_0_22.cohort_num >= 24, period] = df1.loc[df1.cohort_num >= 24, 'real_return_for_ages_0_22']

    return


  def calc_inheritance_from_ages_0_22(self, period: int):
    """
    From 'Lifecycle strategy' sheet.

    Assumes prior steps have been computed for period.

    Starts at row 1292
    """

    # =E1315*(1+F1415)
    self.inheritance_from_ages_0_22.loc[self.inheritance_from_ages_0_22.cohort_num >= 24, period] = self.inheritance_from_ages_0_22.loc[self.inheritance_from_ages_0_22.cohort_num >= 24, period-1] * (1 + self.real_return_for_ages_0_22.loc[self.real_return_for_ages_0_22.cohort_num >= 24, period])
    
    return
      

def initialize_cohort_table_birth():
  """
  From 'Lifecycle strategy' sheet.

  """

  df = pd.DataFrame({'cohort_num': list(range(1, 97)),
                    'born': list(range(1848, 1848 + 96)),
                    'age_22': list(range(1870, 1870 + 96))})

  df1 = pd.DataFrame({key:np.zeros(96) for key in range(0,529)})

  df2 = pd.concat([df, df1], axis=1)

  return df2




def Amount_in_stock(cohort: pd.DataFrame, 
                    retirement_savings_before_period: pd.DataFrame,
                    percentage_target: pd.DataFrame):
    pass
def Present_value_of_accumulation(cohort: pd.DataFrame, 
                                     ):
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


def PE_10_depending_on_period(data_month: pd.DataFrame,
                              pe_10: pd.DataFrame):

  """
  PE Depending on period from 'Lifecycle Investing' sheet, cell F1093

  Static table
  """

  data_month_melted = pd.melt(data_month, 
                              id_vars=['cohort_num', 'begins_work', 'retire'],
                              var_name='period_num',
                              value_name='month')

  df1 = pd.merge(data_month_melted,
                 pe_10.loc[:, ['period_num', 'PE_10']],
                 left_on='month',
                 right_on='period_num',
                 how='left')
  
  df2 = df1.pivot(index=['cohort_num', 'begins_work', 'retire'], 
                  columns='period_num_x',
                  values='PE_10')
  
  df2.columns.name = None
  df2 = df2.reset_index()

  return df2


def PE_Multiplier(PE_10_depending_on_period: pd.DataFrame,
                  pe_10_samuelson: pd.DataFrame):

  """
  From 'Lifecycle strategy' sheet, cell F1192.

  Static table
  """

  pe_array = pe_10_samuelson.loc[:, ['PE_10', 'alpha_adjusted_samuelson_multiplier']].values
  
  def lookup(val, pe_array):
    
    # find all rows where the 0th column is less than or equal to val
    mult = pe_array[pe_array[:,0] <= val, :]

    # sort rows by the first column
    mult = mult[mult[:,0].argsort()]

    # return the bottom row, last colum
    return mult[-1,1]
  
  # find the Samuelson multiplier for each PE_10 value
  pe_mult = [[lookup(val, pe_array) for val in PE_10_depending_on_period.loc[:, c]] for c in range(1, 529)]

  # turn this PE multiplier value into a dataframe
  pe_multiplier = pd.DataFrame(data=np.array(pe_mult).T,
                                  columns=list(range(1,529)))

  # tack on the three ID columns on the left
  pe_multiplier = pd.concat([PE_10_depending_on_period.loc[:, ['cohort_num', 'begins_work', 'retire']], 
                             pe_multiplier], axis=1)
    
  return pe_multiplier


def Samuelson_Share_for_ages_0_22(lambdaearly: float):
  """
  From 'Lifecycle strategy' sheet

  Static table
  """

  df = pd.DataFrame({'cohort_num': list(range(1, 97)),
                     'born': list(range(1848, 1848 + 96)),
                     'age_22': list(range(1870, 1870 + 96))})

  df1 = pd.DataFrame({key:np.zeros(96) for key in range(0,529)})
  df1.iloc[23:95, :] = lambdaearly  # cohort_num >= 24

  df2 = pd.concat([df, df1], axis=1)

  return df2
