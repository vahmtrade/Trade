import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statics.setting import *


def volume_profile(stock, date_start, bins):

    My_window = stock[date_start:].copy()
    dis = My_window["Close"].max() - My_window["Close"].min()
    step = dis / bins
    My_price = np.arange(My_window["Close"].min(), My_window["Close"].max(), step)
    Vp = [0]
    for i in My_price:
        Vp.append(
            My_window[
                (My_window["Close"] > i) & (My_window["Close"] < i + step)
            ].Volume.sum()
        )
    return (Vp[1:], My_price)


dara = pd.read_excel(
    f"{DB}/Boors_Data/DARA ETF-a.xls",
    parse_dates=["Date"],
    index_col="Date",
)
dara["Change"] = dara["Close"].pct_change() * 100
folad = pd.read_excel(
    f"{DB}/Boors_Data/S Mobarakeh Steel-a.xls",
    parse_dates=["Date"],
    index_col="Date",
)
folad["Change"] = folad["Close"].pct_change() * 100
seshargh = pd.read_excel(
    f"{DB}/Boors_Data/Shargh Cement-a.xls",
    parse_dates=["Date"],
    index_col="Date",
)
seshargh["Change"] = seshargh["Close"].pct_change() * 100
fakhouz = pd.read_excel(
    f"{DB}/Boors_Data/Khouz. Steel-a.xls",
    parse_dates=["Date"],
    index_col="Date",
)
fakhouz["Change"] = fakhouz["Close"].pct_change() * 100
shaspa = pd.read_excel(
    f"{DB}/Boors_Data/Spahan Naft-a.xls",
    parse_dates=["Date"],
    index_col="Date",
)
shaspa["Change"] = shaspa["Close"].pct_change() * 100
vamaden = pd.read_excel(
    f"{DB}/Boors_Data/S Metals & Min.-a.xls",
    parse_dates=["Date"],
    index_col="Date",
)
vamaden["Change"] = vamaden["Close"].pct_change() * 100
tala = pd.read_excel(
    f"{DB}/Boors_Data/tala.xls", parse_dates=["Date"], index_col="Date"
)
tala["Change"] = tala["Close"].pct_change() * 100
(Vp, Price) = volume_profile(folad, "2020", 50)
fig, ax = plt.subplots(figsize=[15, 15])
fig.set_facecolor("red")
ax.barh(Price, Vp, 200)
ax.set_title("Volume_Profile")
plt.show()
