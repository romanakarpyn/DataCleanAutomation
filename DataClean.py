import pandas as pd
import numpy as np
from scipy.stats import shapiro
from IPython.display import display


def get_first_look(df):
    out1 = {}
    out2 = {}
    out1['Shape'] = df.shape
    out1 = pd.DataFrame(out1, index=['Rows', 'Columns'])
    out2['Missing values'] = df.select_dtypes(include=['object']).isnull().sum()
    out2 = pd.DataFrame(out2)
    out2 = out2.append(pd.DataFrame(out2.iloc[:, 0].sum(), index=['Total'], columns=['Missing values']))
    out2.index.name = 'Categorical columns'
    out2.reset_index(inplace=True)
    out3 = df.select_dtypes(include=['int', 'float']).isnull().sum().reset_index()
    out3 = pd.DataFrame(out3)
    out3.rename(columns={'index': 'Numerical columns', 0: 'Missing Values'}, inplace=True)
    out2 = pd.concat([out2, out3], axis=1)
    out2.iloc[-1, -1] = out2.iloc[:, -1].sum()
    out4 = df.sample(5, random_state=1)
    return display(out1, out4, out2)


def basic_statistics(df):
    """
    param df: DataFrame
    return: Dataframe with basic statistics
    """
    out = {}
    df = df.select_dtypes(include=['int', 'float'])
    out['Count'] = df.count()
    out['Min'] = df.min()
    out['Max'] = df.max()
    out['Mean'] = round(df.mean())
    out['Range'] = out['Max'] - out['Min']
    out['Q1'] = df.quantile(0.25)
    out['Q3'] = df.quantile(0.75)
    out['IQR'] = out['Q1'] - out['Q3']
    return pd.DataFrame(out).T


def distribution(df):
    """
    :param df: DataFrame
    :return: DataFrame object displaying Skew, Kurtosis, Shapiro statistic and p-value, Normality of distribution
    """
    df = df.select_dtypes(exclude=['object'])
    out = {}
    results = []
    out['Skew'] = df.skew()
    out['Kurtosis'] = df.kurtosis()
    for col in df.columns:
        result = shapiro(df[col])
        results.append(result)
    out = pd.DataFrame(out)
    out2 = pd.DataFrame(results, index=out.index)
    out = pd.concat([out, out2], axis=1)
    normality = pd.DataFrame(np.where(out2.iloc[:, 1] < 0.05, 'Not normal', 'Normal'),
                             columns=['Normality of distribution'], index=out.index)
    out = pd.concat([out, normality], axis=1)
    return display(out)


def get_outliers(df, column_name):
    """
    :param df: DataFrame
    :param column_name: Column name for which we want to find outliers
    :return: DataFrame
    """
    out = {}
    for column in df.select_dtypes(exclude=['object']).columns:
        third_q, first_q = df[column].quantile(0.75), df[column].quantile(0.25)
        interquartile_range = (third_q - first_q) * 1.5
        outlier_high, outlier_low = interquartile_range + third_q, interquartile_range - first_q
        out_iqr = df[column][(df[column] > outlier_high) | (df[column] < outlier_low)]
        out[column] = out_iqr
    return pd.DataFrame(out.get(column_name), columns=[column_name])
