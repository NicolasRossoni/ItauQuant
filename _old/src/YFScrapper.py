import yfinance as yf 
import pandas as pd

ticker = "BZ=F"  # Brent Crude Oil Futures
brent = yf.download(ticker, start="2014-01-01", end="2025-10-29")

print(brent.head())
print(brent.tail())

brent.to_csv("brent_futures_data.csv")