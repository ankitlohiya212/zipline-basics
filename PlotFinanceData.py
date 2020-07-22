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
import nsepy.symbols
from nsepy import get_history
#Can use nsetools as well using pip install nsetools
# API KEY = My2n5erdkggvGrB2hobZ

    
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

def pullDataNSEpy(stockName, index, from_date, to_date):
    from_date, to_date = from_date.split('/'), to_date.split('/')
    for i in range(len(from_date)):
        from_date[i] = int(from_date[i])
        to_date[i] = int(to_date[i])
    df = get_history(symbol=stockName,start=date(from_date[0],from_date[1],from_date[2]),end=date(to_date[0],to_date[1],to_date[2]), index = index)
    return df

# Pulling data Without internet
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
#plotDataYF('INFY',result_range = '12mo',interval = '1d', MA = 20)
def buy_sell_hold(close_prices):
    rsis = RSI(close_prices)
    bought = 0
    buy_price = 0
    tot_profit = 0
    tot_buys = []
    investment = 0
    profit_percent = 0
    start, time_invested = 0, 0
    for num, rsi in enumerate(rsis):
        if rsi >= 70 and bought and (close_prices[num] - buy_price) > 0:
            avg_price = sum(tot_buys)/len(tot_buys)
            profit = (close_prices[num] - avg_price) * bought
            tot_profit += profit
            time_invested += num - start
            profit_percent = profit/(avg_price*bought)*100
            print("Selling Stock at ", close_prices[num])
            print("% profit = ", profit_percent )
            tot_buys = []
            bought = 0
        elif rsi <= 30:
            bought += 1
            buy_price = close_prices[num]
            start = num
            tot_buys.append(buy_price)
            investment += buy_price
            print("Buying Stock at ", buy_price)
    if bought:
        avg_price = sum(tot_buys)/len(tot_buys)
        profit = (close_prices[num] - avg_price) * bought
        tot_profit += profit
        time_invested += num - start
        profit_percent = profit/(avg_price*bought)*100
        print("Selling Stock at ", close_prices[num])
        print("% profit = ", profit_percent )
        tot_buys = []
        bought = 0
        
    return tot_profit,investment, time_invested

##df = pullData('yesBank',5000)
##closep = df['Close'].values.tolist()
##profit, investment, time_invested = buy_sell_hold(closep)
##tot_profit_percent = profit/investment
##onetime_pp = (closep[-1] - closep[0]) / closep[0]
##print('tot_profit_percent = ', tot_profit_percent)
##print("onetime_pp = ", onetime_pp)
##print("time_invested = " , time_invested)



#plotData('DMART', 20, '2019/01/01', '2020/02/18', index = False)





def pullDataYF(stockName, result_range = '1mo',interval = '15m'):
    df = yf(stockName+'.NS', result_range = result_range,interval = interval).result
    return df
"""
def plotDataYF(stockName, result_range = '1mo',interval = '15m', MA = 10):
    #fig.clf()
    fig = plt.figure()
    df = pullDataYF(stockName, result_range = result_range,interval = interval)
    data = df.reset_index().values
    openp, highp, lowp, closep, vol =  df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    date_time, xdays = [], []
    data[:,0] = mdates.date2num(data[:,0])
    ndays = np.unique(np.trunc(list(data[:,0])), return_index=True)
    data2 = np.hstack([np.arange(data[:,0].size)[:, np.newaxis], data[:,1:]])
    for n in np.arange(len(ndays[0])):
        xdays.append(datetime.date.isoformat(num2date(data[ndays[1],0][n])))

    label1 = str(MA)+ ' SMA'
    df['SMA'] = df['Close'].rolling(MA).mean()
    df['std'] = df['Close'].rolling(MA).std()
    df['Bollinger High'] = df['SMA'] + (df['std']*2)
    df['Bollinger Low'] = df['SMA'] - (df['std']*2)
    SP = len(data[:,0][MA-1:])
    
    ax = plt.subplot2grid((6,4),(1,0), rowspan = 4, colspan = 4)
    ax.set_xticks(data2[ndays[1],0])
    ax.plot(data2[:,0], df['SMA'], label = label1, color = 'black')
    ax.plot(data2[:,0], df['Bollinger High'],color = 'red')
    ax.plot(data2[:,0], df['Bollinger Low'], color = 'green')
    plt.legend(fancybox = True, prop = {'size':7})
    plt.setp(ax.get_xticklabels(), visible = False)
    ax.set_ylabel('Price And Volume')
    ax.set_xlabel('Date')
    candlestick_ohlc(ax, data2, width=0.5, colorup='green', colordown='red')
    print("Candlestick done")

    volmin = 0
    axv = ax.twinx()
    axv.axes.yaxis.set_ticklabels([])
    axv.fill_between(list(data2[:,0]), volmin ,vol, facecolor = "#00ffe8", alpha = 0.5)
    axv.grid(False)
    axv.set_ylim(0,4*max(vol))
    print("Volume done")
    
    ax0 = plt.subplot2grid((6,4),(0,0), sharex =ax , rowspan = 1, colspan = 4)
    rsi = RSI(closep)
    ax0.plot(data2[:,0], rsi)
    ax0.axhline(70, color = 'red')
    ax0.axhline(30, color = 'green')
    ax0.set_yticks([30,70])
    #plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune = 'lower'))
    plt.ylabel('RSI')
    plt.setp(ax0.get_xticklabels(), visible = False)
    print("RSI done")

    ax2 = plt.subplot2grid((6,4),(5,0), sharex =ax , rowspan = 1, colspan = 4)
    nslow = 26
    nfast = 12
    nema = 9
    emaslow, emafast, macd = computeMACD(closep)
    ema9 = ExpMA(macd, nema)
    ax2.plot(data2[:,0], macd)
    ax2.plot(data2[:,0], ema9)
    ax2.fill_between(list(data2[:,0]), macd-ema9, 0, facecolor = "#00ffe8", edgecolor = "#00ffe8" )
    plt.ylabel("MACD")
    ax2.set_xticklabels(xdays, rotation=45, horizontalalignment='right')
    print("MACD done")

    plt.suptitle(stockName )
    plt.subplots_adjust(left = 0.09 ,bottom = 0.16,right= 0.98, top = 0.95, wspace=0.2 , hspace=0)
    plt.margins(0,0)
    plt.show()

"""


"""
df = get_history(symbol= 'INFY',start=date(2019,1,1),end=date(2019,12,31))
df.plot()
plt.show()
"""




"""
def animate(i):
	plotDataYF('INFY',result_range = '7d',interval = '1m', MA = 300)
fig = plt.figure()
ani = animation.FuncAnimation(fig, animate, interval = 60000)
plt.show()
"""




"""
while True:
    Stock = input("Enter Stock Name : ")
    Stock = Stock.upper()
    index = bool(int(input("Is Index? : 0/1 ")))
    if not index :
        symbol_df =  nsepy.symbols.get_symbol_list()
        symbol_list = symbol_df['SYMBOL'].tolist()
        if Stock in symbol_list:
            from_date = input("From Date: YYYY/MM/DD ")
            to_date = input("To : YYYY/MM/DD ")
            mov_avg = int(input("Moving Average : "))
        else:
            print("These are the valid stock names : ", symbol_list)
            continue
        
    else :
        indices_list = nsepy.live.NSE_INDICES
        if Stock in indices_list :
            from_date = input("From Date: YYYY/MM/DD ")
            to_date = input("To : YYYY/MM/DD ")
            mov_avg = int(input("Moving Average : "))
        else:
            print("These are the valid indices names : ", indices_list)
            continue
        
    plotData(Stock, mov_avg, from_date, to_date, index = index)
"""
