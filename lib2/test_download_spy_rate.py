from unittest import TestCase
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
from datetime import datetime


class Test(TestCase):
    def test_download(self):
        # Parameters
        start = "2025-01-01"  # or "2024-01-01", etc.
        end = datetime.today().strftime("%Y-%m-%d")

        # --- Get SPY EOD Price ---
        dd = yf.download("SPY", start=start, end=end)
        print(dd.head())
        spy = dd[[("Close", "SPY")]]

        # --- Get SOFR daily rates (proxy for "box" financing) ---
        # SOFR is the financing benchmark most closely tied to box spreads today
        sofr = web.DataReader("SOFR", "fred", start, end)
        sofr.rename(columns={"SOFR": "SOFR_Rate"}, inplace=True)

        # --- Combine into single daily table ---
        df = spy["Close"].join(sofr, how="left")

        # Forward fill SOFR for weekends/holidays
        df["SOFR_Rate"] = df["SOFR_Rate"].fillna(method="ffill")

        print(df.tail())
        df.to_csv("data/spy_eod_and_box_rate.csv")
        print("\nSaved to spy_eod_and_box_rate.csv")
