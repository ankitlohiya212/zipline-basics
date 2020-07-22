import pandas as pd
from trading_calendars import get_calendar
import nsepy
from nsepy import get_history
from datetime import date
import requests
import io
from PlotFinanceData import pullDataYF


def get_data(stock_name):
    df = pullDataYF(stock_name, result_range = '10y', interval='1d')
    df['Date'] = df.index
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['dividend'] = 0.0
    df['split'] = 1.0
    df['date'] = df.index.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df.date)
    my_cal = get_calendar('XBOM')
    valid_days = my_cal.sessions_in_range(df.date.min(), df.date.max())
    
    val_days = list(valid_days.strftime('%Y-%m-%d').values)
    days_in_data = list(df.index.strftime('%Y-%m-%d').values)
    for day in val_days:
        if day not in days_in_data:
            new_date = pd.to_datetime(day)
            df = df.append(pd.DataFrame(index=[new_date]),sort=True)
            
    df['date'] = df.index.strftime('%Y-%m-%d')
    df = df.sort_index()
    df['date'] = pd.to_datetime(df.date)
    df = df.fillna(method='ffill')
    deleted_rows = df[df.date.isin(valid_days) == False]
    df = df[df.date.isin(valid_days)]

    print("Data Length =",len(df))
    print("Valid Days =", len(valid_days))
    return df

def save_file(stock_name, df):
    file_path = "data/daily/{}.csv".format(stock_name) 
    df.to_csv(file_path, index=False)

def get_stock_in_index(index):
    index = index.replace(' ', '').lower()
    index = index.replace('finservice', 'finance')
    url = 'https://www1.nseindia.com/content/indices/ind_{}list.csv'.format(index)
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return df['Symbol'].tolist()

symbols = get_stock_in_index('NIFTY')
start_date = date(2010,1,1)
end_date = date(2020,1,1)

for symbol in symbols:
    df = get_data(symbol)
    if len(df['close']):
        save_file(symbol, df)
        print('Saved File for :', symbol) 
    else:
        print("Failed to extract : ", symbol)
