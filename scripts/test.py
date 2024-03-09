from importlib import reload
import sys
sys.path.append('..')
sys.path.append('c:\\Users\\Emile\\Documents\\lifecycle_investing')

import lc_investing.lifecycle_strategy
lc_investing.lifecycle_strategy = reload(lc_investing.lifecycle_strategy)

s3 = lc_investing.lifecycle_strategy.Simulation()
s3.calc_retirement_savings_before_period()
print(s3.rsbp.FINAL.describe())
# print(s3.rsbp.iloc[0:2, 3:])
print(s3.utility.iloc[0:2, 4:7])


# print(s.monthly_data.columns.tolist())

# print(s.monthly_data.loc[:, ['Monthly_years','Monthly_nom_margin_rate', 'yearly_annual_margin_rate', 'Fed_funds_rate']].tail())


# from importlib import reload
# import lc_investing.constant_percent_stock
# lc_investing.constant_percent_stock = reload(lc_investing.constant_percent_stock)

# s = lc_investing.constant_percent_stock.Simulation(data_folder='./lc_investing/data/',
                                                   # cap=2,
                                                   # incomemult=2.35217277377134,
                                                   # contrate=0.04,
                                                   # ssreplace=0.0,
                                                   # rmm=0.00213711852838,
                                                   # rfm=0.00211039468707308,
                                                   # lambdacons=0.75)
# s.calc_retirement_savings_before_period()
# print(s.retirement_savings_before_period.FINAL.describe())
