
import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time


class Trader(tpqoa.tpqoa):
    
    def __init__(self, conf_file, instrument, interval, bar_length, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0 
        self.profits = []
        
        #*****************strategy-specific attributes***************************
        self.interval = interval
        #************************************************************************
    
    def get_most_recent(self, days = 5):
        while True:
            time.sleep(10)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1].ffill()
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break
                
    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ")
        
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = self.tick_data.append(df)
        
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()

    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define strategy here ************************
        
        now = datetime.utcnow()
        now = now - timedelta(microseconds = now.microsecond)
        today = now - timedelta(minutes = now.minute, hours = now.hour)
        yesterday = today - timedelta(days = 1)
        past = today - timedelta(days = 5)
        raw = self.get_history(instrument = self.instrument, start = past, end = yesterday,
                               granularity = "D", price = "M", localize = False)

        raw["pp"] = (raw.h + raw.l + raw.c) / 3
        raw["pb"] = (raw.h + raw.l) / 2
        raw["pt"] = 2 * raw.pp - raw.pb
        raw["r1"] = 2 * raw.pp - raw.l
        raw["s1"] = 2 * raw.pp - raw.h
        raw["r2"] = raw.pp + raw.h - raw.l
        raw["s2"] = raw.pp - raw.h + raw.l
        
        df = pd.concat([df, raw], ignore_index=False, axis = 1).ffill()
        x = df[self.instrument]
        
        if (x > df.pp).bool:
            df["position"] = np.where((x > df.pt) & (x < df.pt + self.interval), 1, np.nan)
            df["position"] = np.where((x < df.r1) & (x > df.r1 - self.interval), -1, df["position"])
            df["position"] = np.where((x > df.r1 + self.interval) & (x < df.r1 + 2*self.interval), 1, df["position"])
            df["position"] = np.where((x > df.r2 - self.interval), -1, df["position"])
            df["position"] = np.where((x > df.r2 + self.interval), 0, df["position"])
            
        elif (x < df.pp).bool:
            df["position"] = np.where((x < df.pb) & (x > df.pb - self.interval), -1, df["position"])
            df["position"] = np.where((x < df.s1 + self.interval) & (x > df.s1) , 1, df["position"])
            df["position"] = np.where((x < df.s1 - self.interval) & (x > df.s1 - 2*self.interval), -1, df["position"])
            df["position"] = np.where((x < df.s2 + self.interval) & (x > df.s2 - self.interval), 1, df["position"])
            df["position"] = np.where((x < df.s2 - self.interval), 0, df["position"])
             
        df["position"] = df.position.ffill()
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")  
        
        
        
if __name__ == "__main__":
    trader = Trader(r"C:\Users\Dell\Desktop\Algo_Trading_Project\Oanda_firststeps\oanda.cfg", "EUR_USD", "1min", window = 1, units = 100000)
    trader.get_most_recent()
    trader.stream_data(trader.instrument)
    if trader.position != 0: 
        close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
                                          suppress = True, ret = True) 
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0

