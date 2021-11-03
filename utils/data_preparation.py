import ast
import pandas as pd
import numpy as np


def prepare_diffed(df_raw, df_long, 
                   pct_vars, raw_vars, 
                   sign = True):
    """Calculates the difference in years and returns the sign of diff.
    
    Args:
        df_raw: Data frame with raw variables and loan_id.
        df_long: Long format base.
        pct_vars: Variables given as ratios. Not used for
                  diff calculation.
        raw_vars: Variables given as raw values. Used
                  in diff calculation.
        sign: If True, sign of differeces are given rather 
              than the true difference.
    """
    # Calculate difference of values between years
    df_diff = df_raw.groupby(['loan_id']).diff(-1)
    df_diff = pd.concat([
        df_long['statement_years'],
        df_diff,
        df_long[pct_vars]
    ], axis=1)
    # Drop the first years of values since NA when diffing.
    df_diff = df_diff.dropna()
    if sign:
        # Replace raw difference with the sign of the difference
        df_diff.loc[:, raw_vars] = np.sign(df_diff.loc[:, raw_vars])
    
    return df_diff


def concat_with_mi(df):
    indicators = (df == 0).astype(int)
    indicator_labels = [
        s + "__missing"
        for s in df.columns
    ]
    indicators.columns = indicator_labels
    return pd.concat([df, indicators], axis = 1)


def long_to_wide(df: pd.DataFrame, features: list) -> pd.DataFrame:
    years = df.statement_years
    latest_year = years.max()
    to_replace = {
        **{latest_year: '_latest'}, 
        **{year: '_s' + str(i + 1) for i, year in enumerate(years.iloc[1:])}
    }
    df.loc[:, 'statement_years'] = years.replace(to_replace=to_replace)
    pdf = pd.pivot(
        data=df,
        index='loan_id',
        columns='statement_years',
        values=features
    )
    pdf.columns = ['_'.join(col).strip() for col in pdf.columns.values]
        
    return pdf


def preprocess_lookups(nrows=15000):
    """Read, preprocess, and save lookup data"""
    def str_to_list(s: pd.Series) -> pd.Series:
        for k in range(len(s)):
            try:
                s.iloc[k] = ast.literal_eval(s.iloc[k])
            except (ValueError, SyntaxError):
                s.iloc[k] = s.iloc[k]
        return s
    # Define the fs vars
    fs_vars = fetch_fs_vars()
    lookups = pd.read_csv("./input/submission/lookups_parsed__submission.csv", 
                          nrows=nrows)
    ##
    lookups_ = lookups.loc[(lookups['legal_form'] == 'AB')]
    lookups_subset = lookups_[
        ['statement_years', 'months_covered', 'report_id'] + fs_vars
    ].apply(lambda col: str_to_list(col))
    lookups_ = lookups_subset.dropna()
    lookups_.to_pickle("./input/submission/lookups_converted_fields__submission.df")

def fetch_fs_vars():
    """Returns list with fs vars."""
    return None