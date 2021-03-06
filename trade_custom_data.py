from zipline.api import order, symbol, record, set_benchmark, get_open_orders
import datetime
import pytz
from collections import OrderedDict
import zipline
import pandas as pd
from trading_calendars import get_calendar
import matplotlib.pyplot as plt
from matplotlib import style
from PlotFinanceData import *
from statistics import mean, median, mode
style.use('ggplot')
import time

stock = 'AXISBANK' #Choose any of the nifty stocks here


def process_data(stock_name):
    df = pd.read_csv('data/daily/{}.csv'.format(stock.capitalize())) #Finds the file in the path where save_data_for_ingest.py saves data
    df.index = df['date']
    df = df.drop_duplicates()
    data = OrderedDict()
    data[stock] = df[["low","high","open","close","volume"]]

    panel = pd.Panel(data)
    panel.minor_axis = ["low","high","open","close","volume"]
    panel.major_axis = pd.to_datetime(panel.major_axis).tz_localize(pytz.utc)
    return panel

def initialize(context):
    set_benchmark(symbol(stock))
    context.asset = symbol(stock)
    context.i = 0  

def handle_data(context, data):
    context.i += 1
    avg_days = 20
    if context.i < avg_days:
        record(data=data, context=context)
        return
    short_mavg = data.history(context.asset, 'price', bar_count=avg_days,
                              frequency="1d")
    Mean = short_mavg.mean()
    Std = short_mavg.std()
    bollinger_high = Mean + Std*2
    bollinger_low = Mean - Std*2
    
    curr_price = data.history(context.asset, 'price', bar_count= 1,
                              frequency="1d").mean()
    open_orders = get_open_orders()

    if context.asset not in open_orders:
        if curr_price > bollinger_high :
            order(context.asset, -1) #-1 indicates selling 1 stock
        elif curr_price < bollinger_low:
            order(context.asset, 1)
    record(price=data.current(symbol(stock), 'price'), short_mavg=short_mavg,
           data=data, context=context)
    
def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value')
    plt.legend(loc=0)
    plt.show()


panel = process_data(stock)
start = datetime.datetime(2010,7,16,0,0,0,0,pytz.utc)
end = datetime.datetime(2020,7,14,0,0,0,0,pytz.utc)
nse_cal = get_calendar('XBOM')

perf = zipline.run_algorithm(start=start, end=end, initialize=initialize,
                            trading_calendar=nse_cal, capital_base=10000,
                            handle_data=handle_data, analyze = analyze,
                            data_frequency ='daily', data=panel)
