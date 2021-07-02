
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from scipy.optimize import minimize


import RSI
import MACD


def optimal_strategy(parameters):
    
    start = "2004-01-01"
    end = "2020-06-30"
    symbol = "EURUSD=X"
    tc = 0.00007
    
    # SMA
    tester1 = RSI.RSIBacktester(symbol, int(parameters[0]), int(parameters[1]), int(parameters[2]), start, end, tc)
    tester1.test_strategy()
    
    # Bollinger
    tester2 = MACD.MACDBacktester(symbol,  int(parameters[3]),  int(parameters[4]), int(parameters[5]), start, end, tc)
    tester2.test_strategy()
    
    # Create comb
    comb = tester1.results.loc[:, ["returns", "position"]].copy()
    comb.rename(columns = {"position":"position_RSI"}, inplace = True)
    comb["position_MACD"] = tester2.results.position
    
    comb["position_comb"] = np.sign(comb.position_MACD + comb.position_RSI)
    
    # Backtest
    comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]
    comb.dropna(inplace=True)
    comb["trades"] = comb.position_comb.diff().fillna(0).abs()
    comb.strategy = comb.strategy - comb.trades * tc
    comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
    comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)
    
    return -comb["cstrategy"].iloc[-1]

