from datetime import date
import numpy as np
import datetime
import random
import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.dates import num2date
#import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')

from yahoo_finance_api import YahooFinance as yf
import nsepy
from nsepy import get_history
    
def RSI(prices, n=25):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    down = -seed[seed<0].sum()/n
    up = seed[seed>=0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (n-1)+ upval )/n
        down = (down * (n-1)+ downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
    return rsi

def ExpMA(values, window):
	weights = np.exp(np.linspace(-1., 0., window))
	weights/=weights.sum()
	a = np.convolve(values, weights, mode = 'full')[:len(values)]
	a[:window] = a[window]
	return a

def computeMACD(x, slow=26, fast=12):
	emaslow = ExpMA(x, slow)
	emafast = ExpMA(x, fast)
	return emaslow, emafast, emafast - emaslow

#The github repo nsepy could also be used to pull data
def pullDataNSEpy(stockName, index, from_date, to_date):
    from_date, to_date = from_date.split('/'), to_date.split('/')
    for i in range(len(from_date)):
        from_date[i] = int(from_date[i])
        to_date[i] = int(to_date[i])
    df = get_history(symbol=stockName,start=date(from_date[0],from_date[1],from_date[2]),end=date(to_date[0],to_date[1],to_date[2]), index = index)
    return df

#The github repo yahoo_finance_api could also be used to pull data
def pullDataYF(stockName, result_range = '1mo',interval = '15m'):
    df = yf(stockName+'.NS', result_range = result_range,interval = interval).result
    return df

# Pulling data from data directory
def pullData(stock, last_hm_days = 365):
    df = pd.read_csv('data/daily/{}.csv'.format(stock.capitalize()))
    return df[-last_hm_days:]


def plotData(stock, last_hm_days=365):
    #df = pullDataNSEpy(stock, False, '2017/09/01', '2017/10/01')
    #df['Date'] = df.index
    df = pullData(stock, last_hm_days = last_hm_days)
    openp, highp, lowp, closep, vol =  df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values

    days =[]
    for day in df['date'].values:
        days.append(datetime.datetime.strptime(day,'%Y-%m-%d').date())
    df['date'] = days
    
    dates = mdates.date2num(df['date'])
    ohlc = [dates, openp, highp, lowp, closep]
    ohlc = np.transpose(np.array(ohlc))
    MA = int(len(dates)/15)
    label1 = str(MA)+ ' SMA'
    print("Got ohlc")
    df['SMA'] = df['close'].rolling(MA, min_periods = 0).mean()
    df['std'] = df['close'].rolling(MA, min_periods = 0).std()
    df['Bollinger High'] = df['SMA'] + (df['std']*2)
    df['Bollinger Low'] = df['SMA'] - (df['std']*2)
    
    fig = plt.figure()
    
    ax1 = plt.subplot2grid((6,4),(1,0), rowspan = 4, colspan = 4)
    ax1.plot(ohlc[:,0], df['SMA'], color = 'black', label = label1)
    ax1.plot(ohlc[:,0], df['Bollinger High'], color = 'red')
    ax1.plot(ohlc[:,0], df['Bollinger Low'], color = 'green')
    candlestick_ohlc(ax1, ohlc , colorup='green', colordown='red')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(15))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax1.grid(True)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune = 'upper'))
    plt.ylabel('Stock Price and Volume')
    plt.legend(fancybox = True, prop = {'size':7})
    print("Fig 1 done")
    
    ax0 = plt.subplot2grid((6,4),(0,0), sharex =ax1 , rowspan = 1, colspan = 4)
    rsi = RSI(closep)
    ax0.plot(dates, rsi, color = 'blue')
    ax0.axhline(70, color = 'red')
    ax0.axhline(30, color = 'green')
    ax0.fill_between(dates, rsi, 70, where = (rsi>=70), facecolor = "r")
    ax0.fill_between(list(dates), rsi, 30, where = (rsi<=30), facecolor = "g")
    ax0.set_yticks([30,70])
    #plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune = 'lower'))
    plt.ylabel('RSI')
    print("Fig 0 done")
    
    volmin = 0
    ax1v = ax1.twinx()
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.fill_between(list(dates), volmin ,vol,facecolor = "#00ffe8", alpha=0.5 )
    #ax1v.plot(list(dates),vol,color = "#00ffe8" )
    ax1v.grid(False)
    ax1v.set_ylim(0,4*max(vol))
    
    ax2 = plt.subplot2grid((6,4),(5,0), sharex =ax1 , rowspan = 1, colspan = 4)
    nslow = 26
    nfast = 12
    nema = 9
    emaslow, emafast, macd = computeMACD(closep)
    ema9 = ExpMA(macd, nema)
    ax2.plot(dates, macd)
    ax2.plot(dates, ema9)
    ax2.fill_between(list(ohlc[:,0]), macd-ema9, 0, facecolor = "#00ffe8", edgecolor = "#00ffe8" )
    plt.ylabel("MACD")
    print("Fig 2 done")
    plt.subplots_adjust(left = 0.1 ,bottom = 0.19,right= 0.98, top = 0.95, wspace=0.2 , hspace=0)
    plt.setp(ax0.get_xticklabels(), visible = False)
    plt.setp(ax1.get_xticklabels(), visible = False)

    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)
        
    plt.xlabel('Date')
    plt.suptitle(stock )
    plt.margins(0,0)
    plt.show()


#plotData('shreecem', 650)
