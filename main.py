import os
import re
import time
import argparse
from io import StringIO
from typing import List, Tuple

import requests
import tempfile
from zipfile import ZipFile
from concurrent import futures

import numpy as np
import pandas as pd

import datetime
from dateutil.relativedelta import relativedelta


import riskfolio
import statsmodels.api as sm


# Date range
_today = datetime.date.today()

# Last day of previous month
_test_end = _today.replace(day=1) - datetime.timedelta(days=1)
# 10 months of data before starting from test_end
_test_start = _test_end - relativedelta(months=10)

# Training data ends 2 months before test_start... giving a gap to avoid biases
_train_end = _test_start - relativedelta(months=2)
# Training data is 5 years long
_train_start = _train_end - relativedelta(years=5)

# Tickers of assets
# TODO: dynamically retrieve all asset names and prune based on diversification
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
          'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
          'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']


_ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')


def get_timeseries(symbol: str) -> pd.DataFrame:
    api_key = _ALPHA_VANTAGE_API_KEY
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=csv'

    def date_parser(x: pd.Series) -> pd.Series:
        return pd.to_datetime(x, format='%Y-%m-%d').replace(day=1)

    df = pd.read_csv(url, index_col='timestamp', parse_dates=True, date_parser=date_parser)
    df.index.name = 'date'
    df = df.rename(columns={'adjusted close': symbol})

    return df[symbol]


def build_dataset(symbols: List[str]) -> pd.DataFrame:
    # TODO: use multi threading for downloading CSVs
    # TODO: Fill missing values with the previous available value
    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_timeseries, symbols)
        df = pd.concat(results, axis='columns')
        df = df.fillna(method='ffill')
        return df.sort_index()


def get_famafrench() -> pd.DataFrame:
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'
    response = requests.get(url)

    temp_file = tempfile.TemporaryFile()
    temp_file.write(response.content)
    zip_file = ZipFile(temp_file, "r")

    data = zip_file.open(zip_file.namelist()[0]).read().decode()

    tables = [chunk for chunk in data.split(2 * "\r\n") if len(chunk) >= 800]

    source = tables[0]
    match = re.search(r"^\s*,", source, re.M)  # the table starts there
    start = 0 if not match else match.start()
    df = pd.read_csv(
        StringIO("date" + source[start:]), index_col='date', parse_dates=True)
    df.index = pd.to_datetime(df.index, format='%Y%m')
    return df


def factor_betas(factors: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    # Add constant
    factors['const'] = 1.0

    # Forecast beta with factors for each to the returns
    def fit_linear_regression(column_item: Tuple[str, pd.Series]) -> pd.Series:
        model_results = sm.OLS(column_item[1], factors).fit()

        return pd.concat([
            pd.Series([column_item[0]], index=['asset']),
            model_results.params.set_axis(
                'coefficient_' + model_results.params.index.str.replace('-', '').str.lower()),
            model_results.bse.set_axis(
                'std_err_' + model_results.bse.index.str.replace('-', '').str.lower())
        ], axis='index')

    executor = futures.ThreadPoolExecutor()
    results = executor.map(fit_linear_regression, data.iteritems())
    df = pd.concat(results, axis='columns').T.set_index('asset')

    def forecast_beta(factor_name: str) -> pd.Series:
        sse = df[f'std_err_{factor_name}'] ** 2
        coefficient_avg = df[f'coefficient_{factor_name}'].mean()
        coefficient_std = df[f'coefficient_{factor_name}'].std() ** 2

        return sse / (sse + coefficient_std) * coefficient_avg + coefficient_std / (sse + coefficient_std) * df[f'coefficient_{factor_name}']

    executor = futures.ThreadPoolExecutor()
    results = executor.map(forecast_beta, factors.columns)
    beta_forecast = pd.concat(results, axis='columns')
    return beta_forecast



def main() -> None:
    # Downloading data
    print('downloading data')
    # , 'MSFT', 'VZ', 'CNP', 'GOOG'
    data = build_dataset(['AAPL', 'CNP', 'GOOG'])

    print('Calculating returns')
    data = data.pct_change(fill_method='ffill').dropna().loc['2015':'2021']
    print(data.head())

    # Build covariance matrix
    # TODO: find a way to dynamically add more factors
    factors = get_famafrench().loc['2015':'2021']

    factor_betas = factor_betas(factors, data)
    print(factor_betas)

    # TODO: Create covariance matrix with returns + factor betas



    # Building the portfolio object
    # portfolio = riskfolio.HCPortfolio(returns=data.dropna())

    # TODO: get risk free rate from fed
    # # Estimate optimal portfolio:
    # weight = portfolio.optimization(model='HERC', # Could be HRP or HERC
    #                     correlation='pearson', # Correlation matrix used to group assets in clusters,
    #                     rm='MV',  # Risk measure used, this time will be variance
    #                     rf=0,  # Risk free rate,
    #                     linkage='ward', # Linkage method used to build clusters
    #                     max_k=10, # Max number of clusters used in two difference gap statistic
    #                     leaf_order=True)  # Consider optimal order of leafs in dendrogram

    # print(weight)

    # print(data)
    # TODO: Calculate risk metrics


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
