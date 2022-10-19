import numpy as np
import pandas as pd

def create_margin_call_info(data_folder:str,
                            Requirement: float = 0.0,
                            stockadj: float = 0.0,
                            cap: float = 2.0,
                            ):
    """Create margin call information"""

    base_file = data_folder + 'margin_call_info.csv'

    df = pd.read_csv(base_file)

    df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'])
    df.loc[:, 'Close_lead1'] = df.loc[:,'Close'].shift(-1)
    df.loc[:, 'Prospective_Daily_Change'] = df.loc[:,'Close_lead1']/df.loc[:,'Close'] - 1
    df.loc[:, 'Prospective_Daily_Change_plus_one'] = df.loc[:,'Prospective_Daily_Change'] + 1
    df.loc[:, 'Day_of_Month'] = df.loc[:,'Date'].dt.day
    df.loc[df.shape[0]-1, 'Day_of_Month'] = np.nan

    dom = df['Day_of_Month'].tolist()
    
    perf_pre = [1]*df.shape[0]
    for i in range(1, len(perf_pre)):
      perf_pre[i] = df.loc[:, 'Prospective_Daily_Change_plus_one'][i-1]*perf_pre[i-1] if dom[i] > dom[i-1] else 1
    
    perf_after = [1]*df.shape[0]
    for i in range(len(perf_after)-2, -1, -1):
      perf_after[i] = df.loc[:, 'Prospective_Daily_Change_plus_one'][i+1]*perf_after[i+1] if dom[i+1] > dom[i] else 1
    
    df.loc[:, 'Performance_Pre_Date'] = perf_pre
    df.loc[df.shape[0]-1, 'Performance_Pre_Date'] = np.nan

    df.loc[:, 'Performance_days_afterward'] = perf_after
    df.loc[:, 'Performance_Post_Date'] = df.loc[:,'Prospective_Daily_Change_plus_one'] * df['Performance_days_afterward']
    df.loc[:, 'Month'] = (df.loc[:,'Date'].dt.year - 1871) * 12 + df.loc[:,'Date'].dt.month
    df.loc[df.shape[0]-1, 'Month'] = np.nan

    month = df['Month'].tolist()

    worst_so_far = [1]*df.shape[0]
    for i in range(1, len(perf_pre)):
      worst_so_far[i] = 1 if month[i] != month[i-1] else min(df.loc[:, 'Performance_Pre_Date'][i], worst_so_far[i-1])
    df.loc[:, 'Worst_So_Far_This_Month'] = worst_so_far	
    
    worst_later = [1]*(df.shape[0]-1) + [np.nan]
    for i in range(len(worst_later)-2, -1, -1):
      worst_later[i] = 1 if month[i+1] != month[i] else min(df.loc[:, 'Performance_Pre_Date'][i], worst_later[i+1])
    df.loc[:, 'Worst_Later_This_Month']	= worst_later
    
    df.loc[:, 'Worst_This_Month'] = df.loc[:,['Performance_Pre_Date','Worst_So_Far_This_Month','Worst_Later_This_Month']].min(axis=1)
    
    df1 = pd.DataFrame({'a': np.repeat(0, df.shape[0]),
                        'b': df.loc[:,'Worst_This_Month']**12 + stockadj}).max(axis=1)
    
    df.loc[:, 'Adjusted_Annual_Worst_This_Month'] = df1
    df.loc[:, 'Adjusted_Monthly_Worst'] = df.loc[:,'Adjusted_Annual_Worst_This_Month'] ** (1/12)
    
    df.loc[:, 'Margin_Call_Cutoff_Point'] = 1/(1 - df.loc[:,'Adjusted_Monthly_Worst'] + Requirement*df['Adjusted_Monthly_Worst']) if Requirement != 0 else cap
    
    return df