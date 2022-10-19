import pandas as pd 

def create_cohorts():
    """43-year careers start in 1871"""
    df = pd.DataFrame({
        'cohort_num': list(range(1,97)),
        'begins_work': list(range(1871, 1871+96))
    })

    df.loc[:, 'retire'] = df.loc[:, 'begins_work'] + 43

    return df
