import telebot
import sqlite3
import schedule
import time
import threading
import yfinance as yf
import pandas as pd
import datetime

# --- CONFIG ---
# Hardcoded tokens for immediate Termux deployment
BOT_TOKEN = "8338085484:AAEQB6-YHZuUjtwTIUv0DSBKo_9nYBtjZ08"
CHAT_ID = "7708811819"

bot = telebot.TeleBot(BOT_TOKEN)

tier_a_list = [
    "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "KALYANKJIL.NS", "POWERGRID.NS", 
    "TVSMOTOR.NS", "TATASTEEL.NS", "TECHM.NS", "VEDL.NS", 
    "WIPRO.NS", "ZYDUSLIFE.NS"
]

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY, current_capital REAL, in_trade INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS active_trade (ticker TEXT, entry_price REAL, qty REAL, target_price REAL, stop_loss REAL)''')
    
    # Insert initial ₹35,000 capital if database is brand new
    c.execute("SELECT * FROM portfolio")
    if not c.fetchone():
        c.execute("INSERT INTO portfolio (current_capital, in_trade) VALUES (35000.0, 0)")
    conn.commit()
    conn.close()

# --- THE SCAN ENGINE ---
def run_market_scan(manual=False):
    # 1. Check for weekend (Skip if automatic. Allow if manual.)
    today = datetime.datetime.now().weekday()
    if today >= 5 and not manual:
        return

    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute("SELECT in_trade FROM portfolio WHERE id=1")
    in_trade = c.fetchone()[0]
    conn.close()

    # 2. Block new trades if capital is already deployed (Single-Slot Rule)
    if in_trade == 1:
        if manual:
            bot.send_message(CHAT_ID, "🛑 Scan Skipped: You are already in an active trade. Close it first using /close.")
        return

    bot.send_message(CHAT_ID, "🔍 Scanning Tier-A Stocks for +6%/-3% setups...")
    
    # 3. Scan the Tier-A List
    found_signal = False
    radar_report = []  # NEW: List to hold our missed crossover data
    
    for ticker in tier_a_list:
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df.dropna(inplace=True)
            
            # Calculate all crossovers in the dataset
            df['Crossover'] = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1))
            
            # NEW: Find the most recent crossover date
            last_crosses = df[df['Crossover']].index
            if not last_crosses.empty:
                last_cross_date = last_crosses[-1].date()
                days_ago = (datetime.datetime.now().date() - last_cross_date).days
                
                # Format a nice string based on how long ago it was
                if days_ago == 0:
                    status = "Today! 🚨"
                elif days_ago == 1:
                    status = "Yesterday"
                else:
                    status = f"{days_ago} days ago"
                    
                radar_report.append(f"🔸 **{ticker.replace('.NS', '')}**: {status}")
            else:
                radar_report.append(f"🔸 **{ticker.replace('.NS', '')}**: No recent data")
            
            # Check today's crossover for actual execution
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            if (latest['EMA_9'] > latest['EMA_21']) and (prev['EMA_9'] <= prev['EMA_21']) and (latest['Close'] > latest['SMA_200']):
                target = latest['Close'] * 1.06  
                stop = latest['Close'] * 0.97    
                msg = (f"🚨 **BUY SIGNAL DETECTED** 🚨\n\n"
                       f"**Stock:** {ticker.replace('.NS', '')}\n"
                       f"**CMP:** ₹{latest['Close']:.2f}\n"
                       f"**Target (+6%):** ₹{target:.2f}\n"
                       f"**Stop Loss (-3%):** ₹{stop:.2f}\n\n"
                       f"To log this trade, reply with:\n`/buy {ticker.replace('.NS', '')} {latest['Close']:.2f}`")
                bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
                found_signal = True
                break 
        except Exception as e:
            continue

    # NEW: Send the radar report if no trades were taken
    if not found_signal:
        report_text = "\n".join(radar_report)
        final_msg = (f"💤 No actionable setups right now.\n\n"
                     f"📊 **Radar (Last Crossover Missed):**\n"
                     f"{report_text}\n\n"
                     f"Capital remains parked in LIQUIDCASE.")
        bot.send_message(CHAT_ID, final_msg, parse_mode="Markdown")

# --- TELEGRAM COMMANDS ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    if str(message.chat.id) != CHAT_ID: return
    msg = ("👋 **Welcome to the Swing Trading Bot!**\n\n"
           "I manage your single-slot compounding strategy.\n\n"
           "**Commands:**\n"
           "🔸 /scan - Run a manual market scan\n"
           "🔸 /myportfolio - View capital and active trades\n"
           "🔸 /buy [TICKER] [PRICE] - Log a new entry\n"
           "🔸 /close [PRICE] - Close active trade & update capital")
    bot.reply_to(message, msg, parse_mode="Markdown")

@bot.message_handler(commands=['scan'])
def manual_scan(message):
    if str(message.chat.id) != CHAT_ID: return
    bot.reply_to(message, "⚙️ Initiating manual market scan...")
    run_market_scan(manual=True)

@bot.message_handler(commands=['myportfolio', 'portfolio'])
def show_portfolio(message):
    if str(message.chat.id) != CHAT_ID: return
    
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute("SELECT current_capital, in_trade FROM portfolio WHERE id=1")
    port = c.fetchone()
    
    if port[1] == 1:
        c.execute("SELECT * FROM active_trade")
        trade = c.fetchone()
        status = (f"📈 **ACTIVE TRADE** 📈\n\n"
                  f"**Stock:** {trade[0]}\n"
                  f"**Entry:** ₹{trade[1]:.2f}\n"
                  f"**Target:** ₹{trade[3]:.2f}\n"
                  f"**Stop Loss:** ₹{trade[4]:.2f}\n\n"
                  f"**Total Capital Locked:** ₹{port[0]:.2f}")
    else:
        status = (f"💵 **IDLE CAPITAL** 💵\n\n"
                  f"**Total Capital:** ₹{port[0]:.2f}\n"
                  f"**Status:** Parked in LIQUIDCASE")
        
    bot.reply_to(message, status, parse_mode="Markdown")
    conn.close()

@bot.message_handler(commands=['buy'])
def log_buy(message):
    if str(message.chat.id) != CHAT_ID: return
    try:
        parts = message.text.split()
        ticker = parts[1].upper()
        if not ticker.endswith('.NS'): ticker += '.NS'
        entry_price = float(parts[2])
        
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        c.execute("SELECT current_capital, in_trade FROM portfolio WHERE id=1")
        port = c.fetchone()
        
        if port[1] == 1:
            bot.reply_to(message, "⚠️ You are already in a trade! Close it first.")
            conn.close()
            return
            
        qty = port[0] / entry_price
        target = entry_price * 1.06
        stop = entry_price * 0.97
        
        c.execute("INSERT INTO active_trade VALUES (?, ?, ?, ?, ?)", (ticker, entry_price, qty, target, stop))
        c.execute("UPDATE portfolio SET in_trade = 1 WHERE id=1")
        conn.commit()
        conn.close()
        
        bot.reply_to(message, f"✅ Trade logged! Capital is now locked in {ticker}.")
    except Exception as e:
        bot.reply_to(message, "❌ Invalid format. Use: `/buy RELIANCE 2500`", parse_mode="Markdown")

@bot.message_handler(commands=['close'])
def log_close(message):
    if str(message.chat.id) != CHAT_ID: return
    try:
        exit_price = float(message.text.split()[1])
        
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        c.execute("SELECT in_trade, current_capital FROM portfolio WHERE id=1")
        port = c.fetchone()
        
        if port[0] == 0:
            bot.reply_to(message, "⚠️ You have no active trades to close.")
            conn.close()
            return
            
        c.execute("SELECT * FROM active_trade")
        trade = c.fetchone()
        
        # Calculate PnL and update capital
        pnl = (exit_price - trade[1]) * trade[2]
        new_capital = port[1] + pnl
        
        c.execute("UPDATE portfolio SET current_capital = ?, in_trade = 0 WHERE id=1", (new_capital,))
        c.execute("DELETE FROM active_trade")
        conn.commit()
        conn.close()
        
        result = "🟢 PROFIT" if pnl > 0 else "🔴 LOSS"
        bot.reply_to(message, f"✅ Trade Closed!\n{result}: ₹{pnl:.2f}\nNew Capital: ₹{new_capital:.2f}\n\nCapital is now back in LIQUIDCASE.")
    except Exception as e:
        bot.reply_to(message, "❌ Invalid format. Use: `/close 2650` (Enter the exact price you sold at)", parse_mode="Markdown")

# --- SCHEDULER THREAD ---
def run_scheduler():
    # Automatically run at 3:15 PM every day (Termux local time)
    schedule.every().day.at("15:15").do(lambda: run_market_scan(manual=False))
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- START BOT ---
if __name__ == "__main__":
    init_db()
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    print("Bot is running... Press Ctrl+C to stop.")
    try:
        bot.send_message(CHAT_ID, "🤖 Swing Trading Bot is Online! Initial Capital: ₹35,000")
        bot.infinity_polling()
    except Exception as e:
        print(f"Bot stopped: {e}")

