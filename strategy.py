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
    """
    Fetch weekly OHLCV using yfinance and return a clean DataFrame with columns:
    ['Open','High','Low','Close','Volume'] (single-series per column).
    Defends against MultiIndex columns and different casing.
    """
    df = yf.download(tickers=ticker, period=DATA_PERIOD, interval="1wk", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for ticker {ticker}")

    # If yfinance returns MultiIndex columns (happens when multiple tickers passed),
    # try to extract first ticker block.
    if isinstance(df.columns, pd.MultiIndex):
        # pick the first top-level label (usually the ticker)
        top0 = df.columns.levels[0][0]
        try:
            df = df[top0]
        except Exception:
            # fallback: collapse the MultiIndex by taking the first level values
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

    # Normalize column names case-insensitively
    col_map = {c.lower(): c for c in df.columns}
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(col_map.keys())
    if missing:
        # try common alternate names (e.g., 'adj close')
        # create mapping of lowercase names, check if any required close exists as 'adj close'
        if "adj close" in col_map and "close" not in col_map:
            col_map["close"] = col_map["adj close"]
            missing = required - set(col_map.keys())

    if missing:
        raise ValueError(f"Ticker {ticker}: missing required columns {missing} in fetched data. Columns: {list(df.columns)}")

    # Build new clean DataFrame with canonical column names
    clean = pd.DataFrame(index=pd.to_datetime(df.index))
    clean["Open"] = df[col_map["open"]]
    clean["High"] = df[col_map["high"]]
    clean["Low"] = df[col_map["low"]]
    clean["Close"] = df[col_map["close"]]
    clean["Volume"] = df[col_map["volume"]]

    # Ensure numeric and drop rows with all-NaN closes
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")

    clean = clean.dropna(subset=["Close"], how="all")
    if clean.empty:
        raise ValueError(f"No valid OHLC rows for ticker {ticker} after cleaning.")

    clean = clean.sort_index()
    return clean


def supertrend(df, period=SUPER_PERIOD, multiplier=SUPER_MULT):
    """
    Robust Supertrend implementation operating on a cleaned DataFrame
    that has single-series columns for Open/High/Low/Close/Volume.
    Returns df with 'ST_bool', 'ST_value', 'ATR'.
    """
    df = df.copy()
    if len(df) < 2:
        raise ValueError("Not enough data to compute Supertrend (need >= 2 weekly bars).")

    # ensure numeric series exist
    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataframe.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ATR
    atr_series = atr(df, period)
    atr_series = pd.to_numeric(atr_series, errors="coerce").fillna(method="bfill").fillna(method="ffill")

    # HL2
    hl2 = ((df["High"] + df["Low"]) / 2.0).pipe(pd.to_numeric, errors="coerce").fillna(method="bfill").fillna(method="ffill")

    basic_upper_pd = (hl2 + multiplier * atr_series)
    basic_lower_pd = (hl2 - multiplier * atr_series)

    basic_upper = np.asarray(pd.to_numeric(basic_upper_pd, errors="coerce").fillna(method="bfill").fillna(method="ffill"), dtype=float)
    basic_lower = np.asarray(pd.to_numeric(basic_lower_pd, errors="coerce").fillna(method="bfill").fillna(method="ffill"), dtype=float)

    n = len(df)
    final_upper = np.zeros(n, dtype=float)
    final_lower = np.zeros(n, dtype=float)
    st_bool = np.zeros(n, dtype=bool)
    st_value = np.zeros(n, dtype=float)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    st_bool[0] = True
    st_value[0] = final_lower[0]

    close_vals = np.asarray(pd.to_numeric(df["Close"], errors="coerce").fillna(method="bfill").fillna(method="ffill"), dtype=float)

    for i in range(1, n):
        if (basic_upper[i] < final_upper[i-1]) or (close_vals[i-1] > final_upper[i-1]):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if (basic_lower[i] > final_lower[i-1]) or (close_vals[i-1] < final_lower[i-1]):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if st_bool[i-1] and (close_vals[i] <= final_upper[i]):
            st_bool[i] = False
        elif (not st_bool[i-1]) and (close_vals[i] >= final_lower[i]):
            st_bool[i] = True
        else:
            st_bool[i] = st_bool[i-1]

        st_value[i] = final_lower[i] if st_bool[i] else final_upper[i]

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