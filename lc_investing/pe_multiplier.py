import numpy as np
import pandas as pd

def caclulate_pe_10(data_folder: str):

  """Calculate PE 10"""

  base_file = data_folder + 'pe_10.csv'

  df = pd.read_csv(base_file)
  
  # for early months, impute mean
  df.loc[df.Month <= 1880.12, 'PE_10'] = df.loc[((df.Month >= 1881.01) & (df.Month <= 1889.12)), 'PE_10'].mean()
  
  df.insert(loc=0, 
            column='period_num', 
            value=list(range(1, df.shape[0] + 1)))
  
  return df


def caclulate_pe_10_multiplier(data_folder: str,
                              maxsam: float,
                               minsam: float,
                               PEadjust: int):

  """Calculate PE 10 multiplier"""

  base_file = data_folder + 'pe_multiplier.csv'

  df = pd.read_csv(base_file)

  base_value = df.loc[df.Samuelson_Multiplier == 1, 'Samuelson_leveraged'].values[0]
  
  df.loc[:, 'Sam_Multp_with_caps'] = np.maximum(np.minimum(maxsam, df.Samuelson_leveraged), minsam) / base_value
  
  df.loc[:, 'alpha_adjusted_samuelson_multiplier'] = df.loc[:, 'Sam_Multp_with_caps'] if PEadjust==1 else 1

  return df
