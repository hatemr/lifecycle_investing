
import numpy as np
import pandas as pd

def initialize_cohort_table(cohorts):
    """Initialize the cohort table"""

    cohorts_2 = pd.DataFrame(data=np.zeros((cohorts.shape[0], 529)),
                                columns=list(range(529)))

    cohorts_3 = pd.concat([cohorts, cohorts_2], axis=1)

    return cohorts_3


def create_data_month(df_cohort: pd.DataFrame):
    """
    Creates indexes for months.

    Static table
    """
    df1 = df_cohort.copy()

    new_cols = [c + 12*(df1.cohort_num - 1) for c in range(1,529)]

    df2 = pd.concat([df1,
                        pd.DataFrame(np.array(new_cols).T,
                                    columns=list(range(1, len(new_cols)+1)))], axis=1)

    return df2

