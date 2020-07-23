# Zipline For Trading on Custom Data for NSE

First, run the save_data_for_ingest.py file, to store all of the files in a desired directory.
The data is taken using the yahoo_finance_api repo : https://github.com/mayankwadhwa/yahoo_finance_api

Now, you could use this data to ingest in the zipline bundles(which zipline recommends) or use the saved csv files themselves(like I do).

The PlotFinanceData plots all relevant indicators like RSI, MACD, SMA and Bollinger bands on the candlestick chart.

Lastly, the trade_custom_data.py is used to implement a simple Bollinger Band trading strategy: When if price moves above the high Bollinger Band, it should sell, and if the price moves below the lower Bollinger band, it should buy, using the Mean Reversion theory.

