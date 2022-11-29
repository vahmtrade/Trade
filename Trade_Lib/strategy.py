import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from statics.setting import *

plt.style.use("seaborn")


class TesterOneSide:
    def __init__(self, data, start, end, tc, name):
        self.data = data
        self.start = start
        self.end = end
        self.tc = tc
        self.name = name

    def test_strategy(self, sma_s, sma_l, vma_s, vma_l):
        df = self.data["Close"].copy().to_frame()
        df["Value"] = self.data["Value"][self.start : self.end]
        df["SMA_s"] = df["Close"].rolling(sma_s).mean()
        df["SMA_l"] = df["Close"].rolling(sma_l).mean()
        df["VMA_s"] = df["Value"].rolling(vma_s).mean()
        df["VMA_l"] = df["Value"].rolling(vma_l).mean()
        df["Position"] = np.zeros(len(df.index))
        df.dropna(inplace=True)
        df = df[self.start : self.end]
        df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cret"] = df["Ret"].cumsum().apply(np.exp)
        df["Cummax"] = df["Cret"].cummax()
        df["Drow_Down"] = df["Cummax"] - df["Cret"]
        self.df = df
        sig_price_buy = []
        sig_price_sell = []
        flag = -1
        for i in range(len(df.index)):
            if (df["SMA_s"].iloc[i] > df["SMA_l"].iloc[i]) & (
                df["VMA_s"].iloc[i] > df["VMA_l"].iloc[i]
            ):
                if flag != 1:
                    sig_price_buy.append(df["Close"].iloc[i])
                    sig_price_sell.append(np.nan)
                    flag = 1
                    df["Position"].iloc[i] = 1
                else:
                    sig_price_buy.append(np.nan)
                    sig_price_sell.append(np.nan)
                    df["Position"].iloc[i] = 0
            elif df["SMA_s"].iloc[i] < df["SMA_l"].iloc[i]:
                if flag != 0:
                    sig_price_sell.append(df["Close"].iloc[i])
                    sig_price_buy.append(np.nan)
                    df["Position"].iloc[i] = 1
                    flag = 0
                else:
                    sig_price_sell.append(np.nan)
                    sig_price_buy.append(np.nan)
                    df["Position"].iloc[i] = 0
            else:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(np.nan)
                df["Position"].iloc[i] = 0
        df["Sig_Price_Buy"] = sig_price_buy
        df["Sig_Price_Sell"] = sig_price_sell
        trade = df[df["Position"] == 1]
        Close_trade = np.log(trade["Sig_Price_Sell"] / trade["Sig_Price_Buy"].shift(1))
        Close_trade = Close_trade.to_frame()
        Close_trade.rename(columns={0: "Ret"}, inplace=True)
        Close_trade.dropna(inplace=True)
        Close_trade["Cret"] = Close_trade["Ret"].cumsum().apply(np.exp)
        Close_trade["Ret_net"] = Close_trade["Ret"] - self.tc
        Close_trade["Cret_net"] = Close_trade["Ret_net"].cumsum().apply(np.exp)
        Close_trade["Cummax"] = Close_trade["Cret_net"].cummax()
        Close_trade["Drow_Down"] = Close_trade["Cummax"] - Close_trade["Cret_net"]
        self.Close_trade = Close_trade
        self.trade = trade
        return Close_trade["Cret_net"].iloc[-1], df["Cret"].iloc[-1]

    def optimize_strategy(self, range_s, range_l, range_vs, range_vl):
        combination = list(product(range_s, range_l, range_vs, range_vl))
        resault = []
        for c in combination:
            resault.append(self.test_strategy(c[0], c[1], c[2], c[3])[0])
        df = pd.DataFrame(
            columns=["SMA_s", "SMA_l", "VMA_S", "VMA_l"], data=combination
        )
        df["Performance"] = resault
        industry = watchlist[self.name]["indus"]
        df = df.sort_values("Performance", ascending=False)
        df.to_excel(f"{DB}/industries/{industry}/{self.name}/opt.xlsx")
        return df

    def plot_position(self):
        plt.figure(figsize=[20, 8])
        plt.plot(self.df["Close"], alpha=0.25)
        plt.plot(self.df["SMA_s"], alpha=0.5)
        plt.plot(self.df["SMA_l"], alpha=0.5)
        plt.title(f'position of {self.name}')
        plt.scatter(self.df.index, self.df["Sig_Price_Buy"], marker="^", color="green")
        plt.scatter(self.df.index, self.df["Sig_Price_Sell"], marker="v", color="red")
        plt.figure(figsize=[20, 8])
        plt.plot(self.df["Value"], alpha=0.25)
        plt.plot(self.df["VMA_s"], alpha=0.5)
        plt.plot(self.df["VMA_l"], alpha=0.5)
        plt.title(f' Volume position of {self.name}')
        plt.scatter(self.df.index, self.df["Sig_Price_Buy"], marker="^", color="green")
        plt.scatter(self.df.index, self.df["Sig_Price_Sell"], marker="v", color="red")

    def plot_resault(self):
        plt.figure(figsize=[20, 8])
        plt.plot(self.df["Cret"])
        plt.plot(self.Close_trade["Cret_net"], marker="o")
        plt.plot(self.Close_trade["Cret"], marker="o")
        plt.title(f'resault of {self.name}')

class SmaTester:
    def __init__(self, data, start, end, tc):
        self.data = data
        self.start = start
        self.end = end
        self.tc = tc

    def test_strategy(self, sma_s, sma_l):
        df = self.data["Close"][self.start : self.end].to_frame()
        df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cret"] = df["Ret"].cumsum().apply(np.exp)
        df["sma_s"] = df["Close"].rolling(sma_s).mean()
        df["sma_l"] = df["Close"].rolling(sma_l).mean()
        df["Position"] = np.where(df["sma_s"] > df["sma_l"], 1, -1)
        df["Trades"] = df["Position"].diff().fillna(0).abs()
        df["Strategy"] = df["Position"].shift(1) * df["Ret"]
        df["Cstrategy"] = df["Strategy"].cumsum().apply(np.exp)
        df["Strategy_net"] = df["Strategy"] - self.tc * df["Trades"]
        df["Cstrategy_net"] = df["Strategy_net"].cumsum().apply(np.exp)
        self.df = df
        return df["Cstrategy_net"].iloc[-1], df["Cret"].iloc[-1]

    def optimize_parameter(self, range_s, range_l):
        combination = list(product(range_s, range_l))
        resault = []
        for c in combination:
            resault.append(self.test_strategy(c[0], c[1])[0])
        df = pd.DataFrame(data=combination, columns=["SMA_s", "SMA_l"])
        df["Performance"] = resault
        return df

    def plot_position(self, start, end):
        self.df[["Close", "sma_s", "sma_l", "Position"]][start:end].plot(
            figsize=[20, 8], secondary_y="Position"
        )

    def plot_resault(self):
        self.df[["Cstrategy", "Cstrategy_net", "Cret"]].plot(figsize=[20, 8])


class ContrarianTester:
    def __init__(self, data, start, end, tc):
        self.data = data
        self.start = start
        self.end = end
        self.tc = tc

    def test_strategy(self, window):
        df = self.data["Close"].copy().to_frame()
        df = df[self.start : self.end]
        df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cret"] = df["Ret"].cumsum().apply(np.exp)
        df["Cummax"] = df["Cret"].cummax()
        df["Drow_Down"] = df["Cummax"] - df["Cret"]
        df["Position"] = -np.sign(df["Ret"].rolling(window).mean())
        df.dropna(inplace=True)
        df["Trades"] = df["Position"].diff().fillna(0).abs()
        df["Strategy"] = df["Position"].shift(1) * df["Ret"]
        df["Strategy_net"] = df["Strategy"] - df["Trades"] * self.tc
        df["Cstrategy_net"] = df["Strategy_net"].cumsum().apply(np.exp)
        df["Cstrategy"] = df["Strategy"].cumsum().apply(np.exp)
        self.df = df
        self.df = df
        return df["Cstrategy"].iloc[-1], df["Cret"].iloc[-1]

    def optimize_strategy(self, range_w):
        range_w = list(range_w)
        resault = []
        for w in range_w:
            resault.append(self.test_strategy(w)[0])
        perf = pd.DataFrame(columns=["Window", "Performance"])
        perf["Window"] = range_w
        perf["Performance"] = resault
        return perf


class MomentomTester:
    def __init__(self, data, start, end, tc):
        self.data = data
        self.start = start
        self.end = end
        self.tc = tc

    def test_strategy(self, window):
        df = self.data["Close"].copy().to_frame()
        df = df[self.start : self.end]
        df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cret"] = df["Ret"].cumsum().apply(np.exp)
        df["Cummax"] = df["Cret"].cummax()
        df["Drow_Down"] = df["Cummax"] - df["Cret"]
        df["Position"] = np.sign(df["Ret"].rolling(window).mean())
        df.dropna(inplace=True)
        df["Trades"] = df["Position"].diff().fillna(0).abs()
        df["Strategy"] = df["Position"].shift(1) * df["Ret"]
        df["Strategy_net"] = df["Strategy"] - df["Trades"] * self.tc
        df["Cstrategy"] = df["Strategy"].cumsum().apply(np.exp)
        df["Cstrategy_net"] = df["Strategy_net"].cumsum().apply(np.exp)
        self.df = df
        return df["Cstrategy"].iloc[-1], df["Cret"].iloc[-1]

    def optimize_strategy(self, range_w):
        range_w = list(range_w)
        resault = []
        for w in range_w:
            resault.append(self.test_strategy(w)[0])
        perf = pd.DataFrame(columns=["Window", "Performance"])
        perf["Window"] = range_w
        perf["Performance"] = resault
        return perf


class VolumeSma:
    def __init__(self, data, start, end, tc):
        self.data = data
        self.start = start
        self.end = end
        self.tc = tc

    def test_stratgy(self, sma_s, sma_l):
        df = self.data["Close"][self.start : self.end].to_frame()
        df["Value"] = self.data["Value"][self.start : self.end]
        df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cret"] = df["Ret"].cumsum().apply(np.exp)
        df["sma_s"] = df["Value"].rolling(sma_s).mean()
        df["sma_l"] = df["Value"].rolling(sma_l).mean()
        df["Position"] = np.where(df["sma_s"] > df["sma_l"], 1, -1)
        df["Trades"] = df["Position"].diff().fillna(0).abs()
        df["Strategy"] = df["Position"].shift(1) * df["Ret"]
        df["Cstrategy"] = df["Strategy"].cumsum().apply(np.exp)
        df["Strategy_net"] = df["Strategy"] - self.tc * df["Trades"]
        df["Cstrategy_net"] = df["Strategy_net"].cumsum().apply(np.exp)
        self.df = df
        return df["Cstrategy_net"].iloc[-1], df["Cret"].iloc[-1]


class TesterOneSidePrice:
    def __init__(self, data, start, end, tc):
        self.data = data
        self.start = start
        self.end = end
        self.tc = tc

    def test_strategy(self, sma_s, sma_l):
        df = self.data["Close"].copy().to_frame()
        df["SMA_s"] = df["Close"].rolling(sma_s).mean()
        df["SMA_l"] = df["Close"].rolling(sma_l).mean()
        df["Position"] = np.zeros(len(df.index))
        df.dropna(inplace=True)
        df = df[self.start : self.end]
        df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cret"] = df["Ret"].cumsum().apply(np.exp)
        df["Cummax"] = df["Cret"].cummax()
        df["Drow_Down"] = df["Cummax"] - df["Cret"]
        self.df = df
        sig_price_buy = []
        sig_price_sell = []
        flag = -1
        for i in range(len(df.index)):
            if df["SMA_s"].iloc[i] > df["SMA_l"].iloc[i]:

                if flag != 1:
                    sig_price_buy.append(df["Close"].iloc[i])
                    sig_price_sell.append(np.nan)
                    flag = 1
                    df["Position"].iloc[i] = 1
                else:
                    sig_price_buy.append(np.nan)
                    sig_price_sell.append(np.nan)
                    df["Position"].iloc[i] = 0
            elif df["SMA_s"].iloc[i] < df["SMA_l"].iloc[i]:
                if flag != 0:
                    sig_price_sell.append(df["Close"].iloc[i])
                    sig_price_buy.append(np.nan)
                    df["Position"].iloc[i] = 1
                    flag = 0
                else:
                    sig_price_sell.append(np.nan)
                    sig_price_buy.append(np.nan)
                    df["Position"].iloc[i] = 0
            else:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(np.nan)
                df["Position"].iloc[i] = 0
        df["Sig_Price_Buy"] = sig_price_buy
        df["Sig_Price_Sell"] = sig_price_sell
        trade = df[df["Position"] == 1]
        Close_trade = np.log(trade["Sig_Price_Sell"] / trade["Sig_Price_Buy"].shift(1))
        Close_trade = Close_trade.to_frame()
        Close_trade.rename(columns={0: "Ret"}, inplace=True)
        Close_trade.dropna(inplace=True)
        Close_trade["Cret"] = Close_trade["Ret"].cumsum().apply(np.exp)
        Close_trade["Ret_net"] = Close_trade["Ret"] - self.tc
        Close_trade["Cret_net"] = Close_trade["Ret_net"].cumsum().apply(np.exp)
        Close_trade["Cummax"] = Close_trade["Cret_net"].cummax()
        Close_trade["Drow_Down"] = Close_trade["Cummax"] - Close_trade["Cret_net"]
        self.Close_trade = Close_trade
        self.trade = trade
        return Close_trade["Cret_net"].iloc[-1], df["Cret"].iloc[-1]

    def optimize_strategy(self, range_s, range_l):
        combination = list(product(range_s, range_l))
        resault = []
        for c in combination:
            resault.append(self.test_strategy(c[0], c[1])[0])
        df = pd.DataFrame(columns=["SMA_s", "SMA_l"], data=combination)
        df["Performance"] = resault
        return df

    def plot_position(self):
        plt.figure(figsize=[20, 8])
        plt.plot(self.df["Close"], alpha=0.25)
        plt.plot(self.df["SMA_s"], alpha=0.5)
        plt.plot(self.df["SMA_l"], alpha=0.5)
        plt.scatter(self.df.index, self.df["Sig_Price_Buy"], marker="^", color="green")
        plt.scatter(self.df.index, self.df["Sig_Price_Sell"], marker="v", color="red")

    def plot_resault(self):
        plt.figure(figsize=[20, 8])
        plt.plot(self.df["Cret"])
        plt.plot(self.Close_trade["Cret_net"], marker="o")
        plt.plot(self.Close_trade["Cret"], marker="o")
