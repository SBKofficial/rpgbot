#!/usr/bin/env python3
import os
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# ------------------- CONFIGURATIONS -------------------

NIFTY_TICKER = "^NSEI"  # adjust if needed for yfinance data

EQUITY_ETFS = [
    "NIFTYBEES.NS",
    "MOMENTUM.NS",
    "MON100.NS",
    "HDFCSML250.NS",
    "MID150BEES.NS"
]

GOLD_SILVER = [
    "GOLDBEES.NS",
    "SILVERBEES.NS"
]

LIQUID_TICKER = "LIQUIDBEES.NS"

SUPER_PERIOD = 10
SUPER_MULT = 2.5
DATA_PERIOD = "5y"

OUTPUT_FILE = "strategy_summary.json"

# ------------------- DATA FETCH -------------------

def fetch_weekly(ticker):
    df = yf.download(
        tickers=ticker,
        period=DATA_PERIOD,
        interval="1wk",
        progress=False
    )
    if df is None or df.empty:
        raise ValueError(f"No data for ticker {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# ------------------- SUPER TREND -------------------

def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=1).mean()

def supertrend(df, period=SUPER_PERIOD, multiplier=SUPER_MULT):
    """
    Compute Supertrend robustly using scalar access to avoid ambiguous Series truth-values.
    Returns a dataframe with columns 'ST_bool' (bool) and 'ST_value' (float).
    """
    df = df.copy()
    if len(df) < 2:
        raise ValueError("Not enough data to compute Supertrend (need >= 2 weekly bars).")

    atr_series = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2.0

    basic_upper = (hl2 + multiplier * atr_series).values
    basic_lower = (hl2 - multiplier * atr_series).values

    n = len(df)
    final_upper = np.zeros(n, dtype=float)
    final_lower = np.zeros(n, dtype=float)
    st_bool = np.zeros(n, dtype=bool)
    st_value = np.zeros(n, dtype=float)

    # initialize first values
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    st_bool[0] = True  # start bullish by default
    st_value[0] = final_lower[0]

    close_vals = df["Close"].values
    high_vals = df["High"].values
    low_vals = df["Low"].values

    for i in range(1, n):
        # final upper band
        if (basic_upper[i] < final_upper[i-1]) or (close_vals[i-1] > final_upper[i-1]):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # final lower band
        if (basic_lower[i] > final_lower[i-1]) or (close_vals[i-1] < final_lower[i-1]):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # determine trend
        if st_bool[i-1] and (close_vals[i] <= final_upper[i]):
            st_bool[i] = False
        elif (not st_bool[i-1]) and (close_vals[i] >= final_lower[i]):
            st_bool[i] = True
        else:
            st_bool[i] = st_bool[i-1]

        st_value[i] = final_lower[i] if st_bool[i] else final_upper[i]

    # attach results to dataframe
    df["ST_bool"] = st_bool.tolist()
    df["ST_value"] = st_value.tolist()
    df["ATR"] = atr_series.values
    return df

# ------------------- STRATEGY ENGINE -------------------

def analyze():
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "nifty": {},
        "equity": {},
        "gold_silver": {},
        "liquid": LIQUID_TICKER,
        "actions": []
    }

    # Nifty Supertrend
    nifty_df = fetch_weekly(NIFTY_TICKER)
    nifty_st = supertrend(nifty_df)
    nifty_green = bool(nifty_st.iloc[-1]["ST_bool"])

    report["nifty"] = {
        "close": float(nifty_st.iloc[-1]["Close"]),
        "supertrend_green": nifty_green
    }

    # Equity ETFs (Nifty filter applies)
    for etf in EQUITY_ETFS:
        try:
            df = fetch_weekly(etf)
            st = supertrend(df)
            latest = st.iloc[-1]
            prev = st.iloc[-2]

            is_green = bool(latest["ST_bool"])
            was_green = bool(prev["ST_bool"])

            report["equity"][etf] = {
                "close": float(latest["Close"]),
                "st_green": is_green
            }

            if not nifty_green:
                report["actions"].append({
                    "ticker": etf,
                    "action": "SELL",
                    "reason": "NIFTY Weekly Supertrend is RED → exit equities"
                })
            else:
                if is_green and not was_green:
                    report["actions"].append({
                        "ticker": etf,
                        "action": "BUY",
                        "reason": "ETF Supertrend turned GREEN and NIFTY is GREEN"
                    })
                elif not is_green and was_green:
                    report["actions"].append({
                        "ticker": etf,
                        "action": "SELL",
                        "reason": "ETF Supertrend turned RED"
                    })
                else:
                    report["actions"].append({
                        "ticker": etf,
                        "action": "HOLD",
                        "reason": "No change in ETF Supertrend"
                    })

        except Exception as e:
            report["equity"][etf] = {"error": str(e)}
            report["actions"].append({
                "ticker": etf,
                "action": "ERROR",
                "reason": str(e)
            })

    # Gold & Silver (Independent)
    for metal in GOLD_SILVER:
        try:
            df = fetch_weekly(metal)
            st = supertrend(df)
            latest = st.iloc[-1]
            prev = st.iloc[-2]

            is_green = bool(latest["ST_bool"])
            was_green = bool(prev["ST_bool"])

            report["gold_silver"][metal] = {
                "close": float(latest["Close"]),
                "st_green": is_green
            }

            if is_green and not was_green:
                report["actions"].append({
                    "ticker": metal,
                    "action": "BUY",
                    "reason": "Supertrend turned GREEN (independent)"
                })
            elif not is_green and was_green:
                report["actions"].append({
                    "ticker": metal,
                    "action": "SELL",
                    "reason": "Supertrend turned RED"
                })
            else:
                report["actions"].append({
                    "ticker": metal,
                    "action": "HOLD",
                    "reason": "No change in Supertrend"
                })

        except Exception as e:
            report["gold_silver"][metal] = {"error": str(e)}
            report["actions"].append({
                "ticker": metal,
                "action": "ERROR",
                "reason": str(e)
            })

    # LiquidBees rule
    if not nifty_green:
        report["actions"].append({
            "ticker": LIQUID_TICKER,
            "action": "PARK",
            "reason": "NIFTY is RED → move equity capital to LiquidBees"
        })
    else:
        report["actions"].append({
            "ticker": LIQUID_TICKER,
            "action": "STANDBY",
            "reason": "NIFTY is GREEN → LiquidBees unused"
        })

    return report

# ------------------- MAIN -------------------

def main():
    summary = analyze()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()