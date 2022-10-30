from importlib import reload
import lc_investing.lifecycle_strategy
lc_investing.lifecycle_strategy = reload(lc_investing.lifecycle_strategy)

s = lc_investing.lifecycle_strategy.Simulation()
s.calc_retirement_savings_before_period()
print(s.retirement_savings_before_period.FINAL.describe())

# print(s.monthly_data.columns.tolist())

print(s.monthly_data.loc[:, ['Monthly_years','Monthly_nom_margin_rate', 'yearly_annual_margin_rate', 'Fed_funds_rate']].tail())