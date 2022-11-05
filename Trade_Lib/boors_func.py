from cmath import tan
from itertools import cycle
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytse_client as tse
import sklearn.linear_model as linear
import sklearn.metrics as met
import tse_index
import seaborn as sns
import pyswarm as ps
import random
from persiantools.jdatetime import JalaliDate
from statsmodels.tsa.filters import hp_filter
from Trade_Lib.strategy import SmaTester, TesterOneSide, TesterOneSidePrice
from statics.setting import DB, watchlist, regex_en_timeid_q
from Trade_Lib.basic_modules import to_digits
from scipy import stats
import pickle

plt.style.use("seaborn")

def load_stock_analyse(stock_name,name):
    '''
    load your analyse from stock_name/analyse/name.pkl
    '''
    indus=watchlist[stock_name]['indus']
    with open(f'{DB}/industries/{indus}/{stock_name}/analyse/{name}.pkl','rb') as f:
        data=pickle.load(f)
    return data    

def show_analyze(stock_name):
    indus=watchlist[stock_name]['indus']
    for i in os.listdir(f'{DB}/industries/{indus}/{stock_name}/analyse'):
        print(i)


def random_plot():
    number = 200
    chart = []
    for i in range(100000):
        number += random.choice([1, -1])
        chart.append(number)
    plt.figure(figsize=[20, 8])
    plt.title("random_chart")
    plt.plot(chart)
    plt.grid()


def voloume_profile(stock, date_start, bins):

    my_window = stock[date_start:].copy()
    dis = my_window["Close"].max() - my_window["Close"].min()
    step = dis / bins
    my_price = np.arange(my_window["Close"].min(), my_window["Close"].max(), step)
    vp = [0]
    for i in my_price:
        vp.append(
            my_window[
                (my_window["Close"] > i) & (my_window["Close"] < i + step)
            ].Value.sum()
        )
    return (vp[1:], my_price)


def type_record(symbol):
    tse_record = tse.download_client_types_records(symbol)[symbol].set_index("date")
    tse_record.drop("individual_ownership_change", inplace=True, axis=1)
    tse_record.dropna(inplace=True)
    b = []
    for i in tse_record.columns:
        for j in tse_record[i]:
            b.append(int(j))
        c = np.array(b)
        b = []
        tse_record[i] = c
    tse_record["individual_per_cap_buy"] = tse_record["individual_buy_value"] / (
        tse_record["individual_buy_count"]
    )
    tse_record["individual_per_cap_sell"] = tse_record["individual_sell_value"] / (
        tse_record["individual_sell_count"]
    )
    tse_record["individual_per_cap_buy"] = (
        tse_record["individual_per_cap_buy"] / 10000000
    )
    tse_record["individual_per_cap_sell"] = (
        tse_record["individual_per_cap_sell"] / 10000000
    )
    tse_record["buy_power"] = (
        tse_record["individual_per_cap_buy"] / tse_record["individual_per_cap_sell"]
    )
    return tse_record


def read_pe(sma, dev, start, end):
    """
    /fundamental/macro/macro.xlsx"
    """
    pe = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=1,
    )
    pe.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    pe.set_index("Date", inplace=True)
    pe = pe[start:end]
    pe["Rate"] = 1 / pe["Close"]
    pe_mean = pe["Close"][start:end].mean()
    pe_std = pe["Close"][start:end].std()
    pe1 = pe_mean + pe_std
    pe_1 = pe_mean - pe_std
    pe2 = pe_mean + 2 * pe_std
    pe_2 = pe_mean - 2 * pe_std
    pe3 = pe_mean + 3 * pe_std
    pe_3 = pe_mean - 3 * pe_std
    pe_expect = (5 + pe_mean) / 2
    pe["SMA"] = pe["Close"].rolling(sma).mean()
    pe["Upper"] = pe["SMA"] + dev * pe["Close"].rolling(20).std()
    pe["Lower"] = pe["SMA"] - dev * pe["Close"].rolling(20).std()
    P_less_5 = len(pe[pe["Close"] < 5]) / len(pe["Close"])
    p_normal = len(pe[(pe["Close"] > 5) & (pe["Close"] < pe_mean)]) / len(pe["Close"])
    P_hot = len(pe[(pe["Close"] > pe_mean) & (pe["Close"] < pe1)]) / len(pe["Close"])
    P_vhot = len(pe[(pe["Close"] > pe1)]) / len(pe["Close"])
    return pe, [P_less_5, p_normal, P_hot, P_vhot]


def read_pd(start, end):
    p_d = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=2,
    )
    p_d.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    p_d.set_index("Date", inplace=True)
    p_d = p_d[start:end]
    p_d.dropna(inplace=True)
    return p_d


def read_dollar(start, end):
    dollar_azad = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=5,
    )
    dollar_azad.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    dollar_nima = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=4,
    )
    dollar_nima.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    dollar_azad.set_index("Date", inplace=True)
    dollar_nima.set_index("Date", inplace=True)
    dollar_azad = dollar_azad[start:end]
    dollar_nima = dollar_nima[start:end]
    dollar_azad["Change"] = dollar_azad["Close"].pct_change()
    dollar_azad["Ret"] = np.log(dollar_azad["Close"] / dollar_azad["Close"].shift(1))
    dollar_azad["Cret"] = dollar_azad["Ret"].cumsum().apply(np.exp)
    dollar_nima["Change"] = dollar_nima["Close"].pct_change()
    dollar_nima["Ret"] = np.log(dollar_nima["Close"] / dollar_nima["Close"].shift(1))
    dollar_nima["Cret"] = dollar_nima["Ret"].cumsum().apply(np.exp)
    dollar_azad.dropna(inplace=True)
    dollar_nima.dropna(inplace=True)
    return dollar_azad, dollar_nima


def read_ir():
    IR = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=6,
    )
    IR.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    IR.set_index("Date", inplace=True)
    IR.dropna(inplace=True)
    # IR_mean = IR["Close"].mean()
    # IR_std = IR["Close"].std()
    # IR1 = IR_mean + IR_std
    # IR_1 = IR_mean - IR_std
    # IR2 = IR_mean + 2 * IR_std
    # IR_2 = IR_mean - 2 * IR_std
    # IR3 = IR_mean + 3 * IR_std
    # IR_3 = IR_mean - 3 * IR_std
    # plt.figure(figsize=[20, 10])
    # plt.subplot(2, 1, 1)
    # plt.axvline(IR_mean, color="black", linestyle="dashed")
    # plt.axvline(IR1, color="green", linestyle="dashed")
    # plt.axvline(IR_1, color="green", linestyle="dashed")
    # plt.axvline(IR2, color="blue", linestyle="dashed")
    # plt.axvline(IR_2, color="blue", linestyle="dashed")
    # plt.axvline(IR3, color="red", linestyle="dashed")
    # plt.axvline(IR_3, color="red", linestyle="dashed")
    # plt.axvline(IR["Close"].iloc[-1], color="red")
    # plt.hist(IR["Close"], edgecolor="black", bins=60)
    # plt.title("Interest_Rate")
    # plt.subplot(2, 1, 2)
    # plt.plot(IR["Close"], linewidth=3)
    # plt.axhline(y=0.2, linestyle="dashed", color="black")
    return IR


def read_direct(start, end, sma_s, sma_l):
    direct = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=3,
    )
    direct.rename(
        columns={
            "تاریخ میلادی": "Date",
            "تاریخ شمسی": "Jalali",
            "خرید": "Buy",
            "فروش": "Sell",
            "خالص خرید": "Pure",
        },
        inplace=True,
    )
    direct.set_index("Date", inplace=True)
    direct["Buy"] = direct["Buy"] / 10**10
    direct["Sell"] = direct["Sell"] / 10**10
    direct["Pure"] = direct["Pure"] / 10**10
    direct["Sma_s"] = direct["Pure"].rolling(sma_s).mean()
    direct["Sma_l"] = direct["Pure"].rolling(sma_l).mean()
    direct.dropna(inplace=True)
    direct = direct[start:end]
    return direct


def read_value(start, end, sma_s, sma_l):
    value = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=0,
    )
    value.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Value"},
        inplace=True,
    )
    value.set_index("Date", inplace=True)
    value.dropna(inplace=True)
    value = value[start:end]
    value["sma_s"] = value["Value"].rolling(sma_s).mean()
    value["sma_l"] = value["Value"].rolling(sma_l).mean()
    value.dropna(inplace=True)
    return value


def read_value_fix(start, end, sma_s, sma_l):
    value = pd.read_excel(
        f"{DB}/macro/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=3,
    )
    value.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Value"},
        inplace=True,
    )
    value.set_index("Date", inplace=True)
    value.dropna(inplace=True)
    value = value[start:end]
    value["sma_s"] = value["Value"].rolling(sma_s).mean()
    value["sma_l"] = value["Value"].rolling(sma_l).mean()
    value.dropna(inplace=True)
    return value


def plot_pe_ir(pe, IR, start_date):
    fig, ax = plt.subplots(figsize=[20, 8])
    ax.plot(pe[start_date:]["Close"], color="blue", label="P/E")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(IR[start_date:]["Close"], color="red", label="I.R")
    ax2.legend(loc="upper right")


def plot_pe_direct(pe, direct, start_date):
    fig, ax = plt.subplots(figsize=[20, 10])
    ax.plot(pe[start_date:]["Close"], color="blue", label="P/E")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(direct[start_date:]["Pure"], color="red", label="Money_Direct")
    ax2.legend(loc="upper right")


def read_stock(name, start_date, end_date):
    stock = tse.download(name, adjust=True)[name]
    stock.drop("yesterday", inplace=True, axis=1)
    stock.set_index("date", inplace=True)
    stock.rename(
        columns={
            "open": "Open",
            "high": "High",
            "value": "Value",
            "close": "Close",
            "low": "Low",
        },
        inplace=True,
    )
    stock = stock[start_date:end_date]
    stock["Change"] = stock["Close"].pct_change()
    stock.dropna(inplace=True)
    stock["Ret"] = np.log(stock["Close"] / stock["Close"].shift(1))
    stock["Cret"] = stock["Ret"].cumsum().apply(np.exp)
    stock["Cummax"] = stock["Cret"].cummax()
    stock["Drow_Down"] = stock["Cummax"] - stock["Cret"]
    stock.dropna(inplace=True)
    Ticker = tse.Ticker(name)
    # shares = Ticker.total_shares
    # stock["Marcket_Cap"] = shares * stock["Close"] / 10**10
    dollar_azad, dollar_nima = read_dollar(start_date, end_date)
    stock_dollar = stock["Close"].copy()
    stock_dollar = stock_dollar.to_frame()
    stock_dollar["Close"] = stock_dollar["Close"] / dollar_azad["Close"]
    stock_dollar["Change"] = stock_dollar["Close"].pct_change()
    stock_dollar["Ret"] = np.log(stock_dollar["Close"] / stock_dollar["Close"].shift(1))
    stock_dollar["Cret"] = stock_dollar["Ret"].cumsum().apply(np.exp)
    stock_dollar["Cummax"] = stock_dollar["Cret"].cummax()
    stock_dollar["Drow_Down"] = stock_dollar["Cummax"] - stock_dollar["Cret"]
    return stock, stock_dollar


def read_portfolio(broker, owner, date, alpha):
    if broker == "Agah":
        adress = f"{DB}/Portfolio/{broker}/{owner}/raw_data/{date}.xlsx"
        Portfolio = pd.read_excel(adress, usecols="C,D,E,G,H,K", engine="openpyxl")
        Portfolio.rename(
            columns={
                Portfolio.columns[0]: "Stock",
                Portfolio.columns[1]: "Industry",
                Portfolio.columns[2]: "Count",
                Portfolio.columns[3]: "Fix_Price",
                Portfolio.columns[4]: "Mean_Price",
                Portfolio.columns[5]: "Last_Price",
            },
            inplace=True,
        )
    if broker == "Bime":
        adress = f"{DB}/Portfolio/{broker}/{owner}/raw_data/{date}.xls"
        Portfolio = pd.read_excel(adress, usecols="A,C,D,E,G,K")
        Portfolio.rename(
            columns={
                Portfolio.columns[0]: "Stock",
                Portfolio.columns[1]: "Count",
                Portfolio.columns[2]: "Last_Price",
                Portfolio.columns[3]: "Mean_Price",
                Portfolio.columns[4]: "Fix_Price",
            },
            inplace=True,
        )
    Portfolio["stock"] = Portfolio["Stock"]
    Portfolio.set_index("Stock", inplace=True)
    Portfolio["Cost"] = Portfolio["Mean_Price"] * Portfolio["Count"] / 10**7
    Portfolio["Value"] = Portfolio["Count"] * Portfolio["Last_Price"] / 10**7
    Portfolio.sort_values("Value", inplace=True, ascending=False)
    Portfolio["Profit"] = Portfolio["Value"] - Portfolio["Cost"]
    Portfolio.loc["Total"] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        Portfolio["Cost"].sum(),
        Portfolio["Value"].sum(),
        Portfolio["Profit"].sum(),
    ]
    Portfolio["Perc_Of_Value"] = (
        Portfolio["Value"] / Portfolio.loc["Total"]["Value"] * 100
    )
    Portfolio["Perc_Of_Cost"] = Portfolio["Cost"] / Portfolio.loc["Total"]["Cost"] * 100
    Portfolio["Cum_Val"] = Portfolio["Perc_Of_Value"].cumsum()
    Portfolio["Perc_Profit"] = (Portfolio["Value"] / Portfolio["Cost"] - 1) * 100
    small = Portfolio[Portfolio["Perc_Of_Cost"] < (alpha)]
    Portfolio.drop(small.index, inplace=True)
    a = []
    g = []
    for i in Portfolio.index:
        try:
            ticker = tse.Ticker(i)
            a.append(ticker.p_e_ratio)
            g.append(ticker.group_p_e_ratio)
        except:
            a.append(np.nan)
            g.append(np.nan)
    Portfolio["PE"] = a
    Portfolio["PE_Group"] = g
    Portfolio.to_excel(f"{DB}/Portfolio/{broker}/{owner}/Process_data/{date}.xlsx")
    return Portfolio


def group_portFolio(Port):
    Group_Port = pd.DataFrame(
        columns=["Cost", "Value", "Count"],
        index=[
            "Oragh",
            "Tala",
            "Felezat",
            "Chemical",
            "Palayeshgah",
            "Bank",
            "Daroo",
            "Ghaza",
            "Ghand",
            "Khodro",
            "Kashi",
        ],
    )
    Group_Port.loc["Oragh"] = [
        Port[(Port["stock"] == "کیان") | (Port["stock"] == "آکورد")]["Cost"].sum(),
        Port[(Port["stock"] == "کیان") | (Port["stock"] == "آکورد")]["Value"].sum(),
        Port[(Port["stock"] == "کیان") | (Port["stock"] == "آکورد")]["Cost"].count(),
    ]
    Group_Port.loc["Tala"] = [Port.loc["طلا"]["Cost"], Port.loc["طلا"]["Value"], 1]
    Group_Port.loc["Felezat"] = [
        Port[Port["Industry"] == "فلزات اساسي"]["Cost"].sum(),
        Port[Port["Industry"] == "فلزات اساسي"]["Value"].sum(),
        Port[Port["Industry"] == "فلزات اساسي"]["Cost"].count(),
    ]
    Group_Port.loc["Khodro"] = [
        Port[Port["Industry"] == "خودرو و ساخت قطعات"]["Cost"].sum(),
        Port[Port["Industry"] == "خودرو و ساخت قطعات"]["Value"].sum(),
        Port[Port["Industry"] == "خودرو و ساخت قطعات"]["Cost"].count(),
    ]
    Group_Port.loc["Daroo"] = [
        Port[Port["Industry"] == "مواد و محصولات دارويي"]["Cost"].sum(),
        Port[Port["Industry"] == "مواد و محصولات دارويي"]["Value"].sum(),
        Port[Port["Industry"] == "مواد و محصولات دارويي"]["Cost"].count(),
    ]
    Group_Port.loc["Ghand"] = [
        Port[Port["Industry"] == "قند و شكر"]["Cost"].sum(),
        Port[Port["Industry"] == "قند و شكر"]["Value"].sum(),
        Port[Port["Industry"] == "قند و شكر"]["Cost"].count(),
    ]
    Group_Port.loc["Zeraat"] = [
        Port[Port["Industry"] == "زراعت و خدمات وابسته"]["Cost"].sum(),
        Port[Port["Industry"] == "زراعت و خدمات وابسته"]["Value"].sum(),
        Port[Port["Industry"] == "زراعت و خدمات وابسته"]["Cost"].count(),
    ]
    Group_Port.loc["Palayeshgah"] = [
        Port[
            (Port["stock"] == "پالایش")
            | (Port["Industry"] == "فراورده هاي نفتي، كك و سوخت هسته اي")
        ]["Cost"].sum(),
        Port[
            (Port["stock"] == "پالایش")
            | (Port["Industry"] == "فراورده هاي نفتي، كك و سوخت هسته اي")
        ]["Value"].sum(),
        Port[
            (Port["stock"] == "پالایش")
            | (Port["Industry"] == "فراورده هاي نفتي، كك و سوخت هسته اي")
        ]["Value"].count(),
    ]
    Group_Port.loc["Chemical"] = [
        Port[Port["Industry"] == "محصولات شيميايي"]["Cost"].sum(),
        Port[Port["Industry"] == "محصولات شيميايي"]["Value"].sum(),
        Port[Port["Industry"] == "محصولات شيميايي"]["Cost"].count(),
    ]
    Group_Port.loc["Mokhaberat"] = [
        Port[Port["Industry"] == "مخابرات"]["Cost"].sum(),
        Port[Port["Industry"] == "مخابرات"]["Value"].sum(),
        Port[Port["Industry"] == "مخابرات"]["Cost"].count(),
    ]
    Group_Port.loc["Sarmaye_Gozari"] = [
        Port[Port["Industry"] == "سرمايه گذاريها"]["Cost"].sum(),
        Port[Port["Industry"] == "سرمايه گذاريها"]["Value"].sum(),
        Port[Port["Industry"] == "سرمايه گذاريها"]["Cost"].count(),
    ]
    Group_Port.loc["Bank"] = [
        Port[
            (Port["stock"] == "دارا یکم")
            | (Port["Industry"] == "بانكها و موسسات اعتباري")
        ]["Cost"].sum(),
        Port[
            (Port["stock"] == "دارا یکم")
            | (Port["Industry"] == "بانكها و موسسات اعتباري")
        ]["Value"].sum(),
        Port[
            (Port["stock"] == "دارا یکم")
            | (Port["Industry"] == "بانكها و موسسات اعتباري")
        ]["Value"].count(),
    ]
    Group_Port.loc["Utility"] = [
        Port[Port["Industry"] == "عرضه برق، گاز، بخاروآب گرم"]["Cost"].sum(),
        Port[Port["Industry"] == "عرضه برق، گاز، بخاروآب گرم"]["Value"].sum(),
        Port[Port["Industry"] == "عرضه برق، گاز، بخاروآب گرم"]["Cost"].count(),
    ]
    Group_Port.loc["Siman"] = [
        Port[Port["Industry"] == "سيمان، آهك و گچ"]["Cost"].sum(),
        Port[Port["Industry"] == "سيمان، آهك و گچ"]["Value"].sum(),
        Port[Port["Industry"] == "سيمان، آهك و گچ"]["Cost"].count(),
    ]
    Group_Port.loc["Kashi"] = [
        Port[Port["Industry"] == "کاشی و سرامیک"]["Cost"].sum(),
        Port[Port["Industry"] == "کاشی و سرامیک"]["Value"].sum(),
        Port[Port["Industry"] == "کاشی و سرامیک"]["Cost"].count(),
    ]
    Group_Port.loc["Ghaza"] = [
        Port[Port["Industry"] == "غذايي بجز قند"]["Cost"].sum(),
        Port[Port["Industry"] == "غذايي بجز قند"]["Value"].sum(),
        Port[Port["Industry"] == "غذايي بجز قند"]["Cost"].count(),
    ]
    Group_Port.loc["Sanati"] = [
        Port[Port["Industry"] == "شرکتهاي چند رشته اي صنعتي"]["Cost"].sum(),
        Port[Port["Industry"] == "شرکتهاي چند رشته اي صنعتي"]["Value"].sum(),
        Port[Port["Industry"] == "شرکتهاي چند رشته اي صنعتي"]["Cost"].count(),
    ]
    Group_Port.loc["Computer"] = [
        Port[Port["Industry"] == "توليد محصولات كامپيوتري الكترونيكي ونوري"][
            "Cost"
        ].sum(),
        Port[Port["Industry"] == "توليد محصولات كامپيوتري الكترونيكي ونوري"][
            "Value"
        ].sum(),
        Port[Port["Industry"] == "توليد محصولات كامپيوتري الكترونيكي ونوري"][
            "Cost"
        ].count(),
    ]
    Group_Port.loc["Total"] = [
        Group_Port["Cost"].sum(),
        Group_Port["Value"].sum(),
        Group_Port["Count"].sum(),
    ]
    Group_Port["Profit"] = Group_Port["Value"] / (Group_Port["Cost"] + 0.01) - 1
    Group_Port["Percent"] = Group_Port["Value"] / Group_Port.loc["Total"]["Value"]
    Group_Port["Cperc"] = Group_Port["Percent"].cumsum()

    return Group_Port


def group_portfolio_bime(Portfolio):
    Group_Port = pd.DataFrame(
        columns=["Count", "Name", "Cost", "Value"],
        index=["Oragh", "Tala", "Bank", "Felezat", "Sarmaye_Gozari"],
    )
    Group_Port["Count"] = [2, 1, 1, 1, 1]
    Group_Port["Name"] = ["Kian,Akord", "Tala", "Dara_Yekom", "Folad", "Vakharazm"]
    Group_Port["Cost"] = [
        Portfolio.iloc[0]["Cost"] + Portfolio.iloc[1]["Cost"],
        Portfolio.loc["طلا"]["Cost"],
        Portfolio.iloc[3]["Cost"],
        Portfolio.loc["فولاد"]["Cost"],
        Portfolio.loc["وخارزم"]["Cost"],
    ]
    Group_Port["Value"] = [
        Portfolio.iloc[0]["Value"] + Portfolio.iloc[1]["Value"],
        Portfolio.loc["طلا"]["Value"],
        Portfolio.iloc[3]["Value"],
        Portfolio.loc["فولاد"]["Value"],
        Portfolio.loc["وخارزم"]["Value"],
    ]
    Group_Port["Perc_Of_Value"] = [
        Portfolio.iloc[0]["Perc_Of_Value"] + Portfolio.iloc[1]["Perc_Of_Value"],
        Portfolio.loc["طلا"]["Perc_Of_Value"],
        Portfolio.iloc[3]["Perc_Of_Value"],
        Portfolio.loc["فولاد"]["Perc_Of_Value"],
        Portfolio.loc["وخارزم"]["Perc_Of_Value"],
    ]
    Group_Port.sort_values("Value", ascending=False, inplace=True)
    Group_Port["Cum_Val"] = Group_Port["Perc_Of_Value"].cumsum()
    plt.figure(figsize=[30, 10])
    Group_Port["Value"].plot(kind="pie", autopct="%1.2f")
    return Group_Port


def history(Broker, owner):
    Data = []
    c = []
    v = []
    History_Portfolio = pd.DataFrame(columns=["Cost", "Value"])
    for i in os.listdir(f"{DB}/Portfolio/Agah/{owner}/Process_data"):
        adress = f"{DB}/Portfolio/{Broker}/{owner}/Process_data/{i}"
        df = pd.read_excel(adress)
        df.set_index("Stock", inplace=True)
        Data.append(df)
    for i in range(len(Data)):
        c.append(Data[i].loc["Total"]["Cost"])
        v.append(Data[i].loc["Total"]["Value"])
    History_Portfolio["Cost"] = c
    History_Portfolio["Value"] = v
    History_Portfolio["Change"] = History_Portfolio["Value"].pct_change() * 100
    plt.figure(figsize=[15, 8])
    plt.plot(History_Portfolio["Value"], marker="o", linewidth=3)
    plt.title("History_of_Portfolio")
    return Data, History_Portfolio


def get_income_yearly(stock, money_type, n):
    industry = watchlist[stock]["indus"]
    if money_type == "rial":
        adress = f"{DB}/industries/{industry}/{stock}/income/yearly/rial.xlsx"
    elif money_type == "dollar":
        adress = f"{DB}/industries/{industry}/{stock}/income/yearly/dollar.xlsx"

    stock_income = pd.read_excel(adress, engine="openpyxl")
    all_time_id = re.findall(regex_en_timeid_q, str(stock_income.loc[6]))
    years=[]
    for i in all_time_id:
        years.append(int(i[:4]))
    year = int(all_time_id[-1][:4])
    fiscal_year = int(all_time_id[-1][5:])
    stock_income = pd.read_excel(adress, engine="openpyxl", usecols="B:end", skiprows=7)
    stock_income.drop("Unnamed: 2", axis=1, inplace=True)
    stock_income.drop([0, 1, 6, 14], inplace=True)
    stock_income.dropna(inplace=True)
    for i in stock_income.index:
        for j in stock_income.columns:
            if stock_income.loc[i][j] == "-":
                stock_income.loc[i][j] = 0.01
            if stock_income.loc[i][j] == "-":
                stock_income.loc[i][j] = 0.1
    my_col = []
    my_col.append("Data")
    n = [year - i for i in range(n)]
    my_col.extend(n[::-1])
    for i in range(len(stock_income.columns)):
        stock_income.rename(columns={stock_income.columns[i]: my_col[i]}, inplace=True)
    stock_income.set_index("Data", inplace=True)

    my_index = [
        "Total_Revenue",
        "Cost_of_Revenue",
        "Gross_Profit",
        "Operating_Expense",
        "Other_operating_Income_Expense",
        "Operating_Income",
        "Interest_Expense",
        "Other_non_operate_Income_Expense",
        "Pretax_Income",
        "Tax_Provision",
        "Net_Income_Common",
        "Net_Profit",
        "EPS",
        "Capital",
        "EPS_Capital",
    ]

    for i in range(len(stock_income.index)):
        stock_income.rename(index={stock_income.index[i]: my_index[i]}, inplace=True)
    for i in stock_income.columns:
        if stock_income.loc["Total_Revenue"][i] == 0.01:
            stock_income.drop(i, axis=1, inplace=True)
    for i in stock_income.columns:
        if stock_income.loc["Total_Revenue"][i] == 0:
            stock_income.loc["Total_Revenue"][i] = 1
    stock_common_size = pd.DataFrame(
        columns=stock_income.columns, index=stock_income.index
    )
    for i in stock_common_size.index:
        stock_common_size.loc[i] = (
            stock_income.loc[i] / stock_income.loc["Total_Revenue"]
        )
    stock_income = stock_income.T
    stock_common_size = stock_common_size.T
    mean_p_m = stock_common_size["Operating_Income"].mean()
    risk_p_m = stock_common_size["Operating_Income"].std()
    a = stock_income["Total_Revenue"].pct_change()
    cagr = a.mean()
    return (
        stock_income,
        stock_common_size,
        [mean_p_m, risk_p_m],
        cagr,
        fiscal_year,
        year,
    )


def get_income_quarterly(stock, money_type, n, fisal_year):
    industry = watchlist[stock]["indus"]
    if money_type == "rial":
        adress = f"{DB}/industries/{industry}/{stock}/income/quarterly/rial.xlsx"
    if money_type == "dollar":
        adress = f"{DB}/industries/{industry}/{stock}/income/quarterly/dollar.xlsx"

    fiscal_dic = {
        12: {3: 1, 6: 2, 9: 3, 12: 4},
        9: {12: 1, 3: 2, 6: 3, 9: 4},
        6: {9: 1, 12: 2, 3: 3, 6: 4},
        3: {6: 1, 9: 2, 12: 3, 3: 4},
        10: {1: 1, 4: 2, 7: 3, 10: 4},
    }
    stock_income = pd.read_excel(adress, engine="openpyxl")
    all_time_id = re.findall(regex_en_timeid_q, str(stock_income))
    my_year = int(all_time_id[-1][:4])
    my_month = int(all_time_id[-1][5:])
    my_Q = fiscal_dic[fisal_year][my_month]
    stock_income = pd.read_excel(adress, skiprows=7, engine="openpyxl")
    stock_income.drop([1, 6, 14], inplace=True)
    stock_income.drop(["Unnamed: 2", "Unnamed: 0"], axis=1, inplace=True)
    # data_Cleaning replace '-'
    for i in stock_income.index:
        for j in stock_income.columns:
            if stock_income.loc[i][j] == "-":
                stock_income.loc[i][j] = 0.01
            if stock_income.loc[i][j] == 0:
                stock_income.loc[i][j] = 0.1
    # data_Cleaning miladi tarikh_enteshar
    date_release = []
    for i in stock_income.iloc[0][1:]:
        i = i[0:10]
        b = [int(j) for j in i.split("-")]
        year = b[0]
        month = b[1]
        day = b[2]
        date_release.append(pd.to_datetime(JalaliDate(year, month, day).to_gregorian()))
    stock_income.iloc[0][1:] = date_release
    my_col = []
    session = []
    year = []
    for i in range(n):
        data = "{} Q_{}".format(my_year, my_Q)
        session.append(my_Q)
        year.append(my_year)
        my_col.append(data)
        my_Q = my_Q - 1
        if my_Q == 0:
            my_Q = 4
            my_year = my_year - 1
    my_col.append("Data")
    my_col = my_col[::-1]
    session = session[::-1]
    year = year[::-1]
    for i in range(len(stock_income.columns)):
        stock_income.rename(columns={stock_income.columns[i]: my_col[i]}, inplace=True)
    stock_income.set_index("Data", inplace=True)
    stock_income.dropna(inplace=True)
    my_index = [
        "Realese_date",
        "Total_Revenue",
        "Cost_of_Revenue",
        "Gross_Profit",
        "Operating_Expense",
        "Other_operating_Income_Expense",
        "Operating_Income",
        "Interest_Expense",
        "Other_non_operating_Income_Expense",
        "Pretax_Income",
        "Tax_Provision",
        "Net_Income_Common",
        "Net_Profit",
        "EPS",
        "Capital",
        "EPS_Capital",
    ]
    for i in range(len(stock_income.index)):
        stock_income.rename(index={stock_income.index[i]: my_index[i]}, inplace=True)
    stock_common_size = pd.DataFrame(
        columns=stock_income.columns, index=stock_income.index
    )
    for i in stock_income.index[1:]:
        stock_common_size.loc[i] = (
            stock_income.loc[i] / stock_income.loc["Total_Revenue"]
        )
    stock_common_size.loc["Realese_date"] = date_release
    stock_income = stock_income.T
    stock_common_size = stock_common_size.T
    stock_income["Session"] = session
    stock_common_size["Session"] = session
    stock_income["Year"] = year
    stock_common_size["Year"] = year
    mean_p_m = stock_common_size["Operating_Income"].mean()
    risk_p_m = stock_common_size["Operating_Income"].std()
    a = stock_income["Total_Revenue"].pct_change()
    cagr = a.mean()
    return stock_income, stock_common_size, [mean_p_m, risk_p_m], cagr


def type_record(nemad):
    tse_record = tse.download_client_types_records(nemad)[nemad].set_index("date")
    tse_record.drop("individual_ownership_change", inplace=True, axis=1)
    tse_record.dropna(inplace=True)
    b = []
    for i in tse_record.columns:
        for j in tse_record[i]:
            b.append(int(j))
        c = np.array(b)
        b = []
        tse_record[i] = c
    tse_record["individual_per_cap_buy"] = tse_record["individual_buy_value"] / (
        tse_record["individual_buy_count"]
    )
    tse_record["individual_per_cap_sell"] = tse_record["individual_sell_value"] / (
        tse_record["individual_sell_count"]
    )
    tse_record["individual_per_cap_buy"] = (
        tse_record["individual_per_cap_buy"] / 10000000
    )
    tse_record["individual_per_cap_sell"] = (
        tse_record["individual_per_cap_sell"] / 10000000
    )
    tse_record["Buy_Power"] = (
        tse_record["individual_per_cap_buy"] / tse_record["individual_per_cap_sell"]
    )
    my_data = tse_record[["Buy_Power"]]
    return my_data


def read_index(Name, my_interval, start, end):
    index = tse_index.reader()
    my_index = index.history(Name, interval=my_interval)
    my_index = my_index[start:end]
    if my_interval == "w":
        my_index.rename(columns={"close": "Close"}, inplace=True)
    my_index["Change"] = my_index["Close"].pct_change()
    my_index["Ret"] = np.log(my_index["Close"] / my_index["Close"].shift(1))
    my_index["Cret"] = my_index["Ret"].cumsum().apply(np.exp)
    my_index["Cummax"] = my_index["Cret"].cummax()
    my_index["Drow_Down"] = my_index["Cummax"] - my_index["Cret"]
    return my_index


def plot_marq(stocks, y_s=1400, m_s=1, d_s=1, y_e=1401, m_e=6, d_e=1):
    start = pd.to_datetime(JalaliDate(y_s, m_s, d_s).to_gregorian())
    end = pd.to_datetime(JalaliDate(y_e, m_e, d_e).to_gregorian())
    stocks_ret = []
    stocks_risk = []
    stocks_name = []
    desc = []
    for i in stocks:
        stocks_name.append(i.Name)
    for i in stocks:
        stocks_ret.append(np.exp(i.Price["Ret"][start:end].sum()))
        stocks_risk.append(np.std(i.Price["Ret"][start:end]))
        desc.append(i.Price[start:end]["Change"].describe())
    Ret_describe = pd.DataFrame(desc, index=stocks_name)
    plt.figure(figsize=[20, 10])
    for i in range(len(stocks_name)):
        plt.annotate(
            stocks_name[i], xy=(stocks_risk[i] + 0.0002, stocks_ret[i] + 0.0002)
        )
    plt.plot(stocks_risk, stocks_ret, "o", markersize=12)
    plt.grid()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Risk_Return")
    return Ret_describe


def plot_margin(stocks):
    stocks_name = []
    margin_mean_yearly = []
    margin_mean_quarterly = []
    margin_risk_yearly = []
    margin_risk_quarterly = []
    for i in stocks:
        stocks_name.append(i.Name)
    for i in stocks:
        margin_mean_yearly.append(i.Risk_income_yearly[0])
        margin_risk_yearly.append(i.Risk_income_yearly[1])
        margin_mean_quarterly.append(i.Risk_income_rial_quarterly[0])
        margin_risk_quarterly.append(i.Risk_income_rial_quarterly[1])
    plt.figure(figsize=[20, 10])
    plt.title("Yearly")
    for i in range(len(stocks_name)):
        plt.annotate(
            stocks_name[i],
            xy=(margin_risk_yearly[i] + 0.0002, margin_mean_yearly[i] + 0.0002),
        )
    plt.plot(margin_risk_yearly, margin_mean_yearly, "o", markersize=12)
    plt.figure(figsize=[20, 10])
    plt.title("quarterly")
    for i in range(len(stocks_name)):
        plt.annotate(
            stocks_name[i],
            xy=(margin_risk_quarterly[i] + 0.0002, margin_mean_quarterly[i] + 0.0002),
        )
    plt.plot(margin_risk_quarterly, margin_mean_quarterly, "o", markersize=12)


def plot_pe_stocks(stocks):
    stocks_name = []
    pe = []
    pe_mean = []
    pe_std = []
    for i in stocks:
        stocks_name.append(i.Name)
    for i in stocks:
        pe.append(i.pe["P/E-ttm"].iloc[0])
        pe_mean.append(i.pe["P/E-ttm"].median())
        pe_std.append(i.pe["P/E-ttm"].std())
    df = pd.DataFrame(columns=["Name"])
    df["Name"] = stocks_name
    df["P/E"] = pe
    df["P/E_mean"] = pe_mean
    df["P/E_std"] = pe_std
    plt.figure(figsize=[15, 8])
    plt.bar(x=df["Name"], height=df["P/E"])
    plt.scatter(x=df["Name"], y=df["P/E_mean"])
    plt.figure(figsize=[15, 8])
    plt.plot(df["P/E_std"], df["P/E_mean"], "o", markersize=12)
    for i in range(len(stocks_name)):
        plt.annotate(
            stocks_name[i],
            xy=(df["P/E_std"][i] + 0.0002, df["P/E_mean"][i] + 0.0002),
        )

    return df


def plot_dollar_analyse(stocks):
    plt.figure(figsize=[20, 8])
    for i in stocks:
        plt.plot(i.dollar_analyse["Total_Revenue"], label=i.Name, marker="o")
    plt.legend()
    plt.title("Total_Rev_Dollar")
    plt.figure(figsize=[16, 8])
    for i in stocks:
        plt.plot(i.dollar_analyse["Net_Profit"], label=i.Name, marker="o")
    plt.legend()
    plt.title("Net_Profit_Dollar")


def plot_stocks_ret(stocks, year_s, month_s, day_s, year_e, month_e, day_e):
    start = pd.to_datetime(JalaliDate(year_s, month_s, day_s).to_gregorian())
    end = pd.to_datetime(JalaliDate(year_e, month_e, day_e).to_gregorian())
    plt.figure(figsize=[20, 8])
    for i in stocks:
        plt.plot(
            i.Price[start:end]["Close"] / i.Price[start:end]["Close"].iloc[0],
            label=i.Name,
        )
        plt.legend()


def plot_cagr_stocks(stocks):
    plt.figure(figsize=[12, 6])
    for i in stocks:
        plt.bar(x=i.Name, height=i.cagr_rial_yearly)


def plot_margin_trend(stocks):
    plt.figure(figsize=[16, 8])
    for i in stocks:
        plt.plot(i.income_common_rial_yearly[["Net_Profit"]], label=i.Name, marker="o")
    plt.legend()
    plt.title("Net_Profit_Margin_yearly")
    plt.figure(figsize=[16, 8])
    for i in stocks:
        plt.plot(i.income_common_rial_quarterly[["Net_Profit"]], label=i.Name, marker="o")
    plt.legend()
    plt.title("Net_Profit_Margin_quarterly")
    plt.figure(figsize=[16, 8])


def plot_pe_trend_stocks(stocks, year_s, month_s, year_e, month_e):
    start = pd.to_datetime(JalaliDate(year_s, month_s, 1).to_gregorian())
    end = pd.to_datetime(JalaliDate(year_e, month_e, 1).to_gregorian())
    plt.figure(figsize=[20, 8])
    for i in stocks:
        plt.plot(i.pe[end:start]["P/E-ttm"], label=i.Name)
    plt.legend()


def plot_revenue_stocks(stocks):
    plt.figure(figsize=[20, 8])


def plot_corr(df):
    cor = df.corr()
    plt.figure(figsize=[20, 12], facecolor="white")
    sns.set(font_scale=1.5)
    sns.heatmap(cor, cmap="Reds", annot=True, annot_kws={"size": 12}, vmax=1, vmin=-1)


def plot_voloume_profile(price, start, n):
    vp, bin = voloume_profile(price, start, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10])
    ax1.barh(bin, vp, n, edgecolor="black")
    ax1.grid(color="white")
    ax2.plot(price["2020":]["Close"], linewidth=3)
    ax2.grid(color="white")


def desired_portfolio(names, farsi, start, end):
    data = pd.DataFrame(columns=[names[0], names[1]])
    a = []
    b = []
    for i in range(len(names)):
        data[names[i]] = read_stock(farsi[i], start, end)[0]["Close"]
    data.dropna(inplace=True)
    for i in range(len(names)):
        col = names[i] + "_ret"
        a.append(col)
        data[col] = np.log(data[names[i]] / data[names[i]].shift(1))
    data.dropna(inplace=True)
    for i in range(len(a)):
        col_2 = names[i] + "_cret"
        b.append(col_2)
        data[col_2] = data[a[i]].cumsum().apply(np.exp)
    return data, b, a


def mix_portfolio(names, prices, start, end):
    data = pd.DataFrame(columns=names)
    a = []
    b = []
    for i in range(len(names)):
        data[names[i]] = prices[i]["Close"][start:end]
    data.dropna(inplace=True)
    for i in range(len(names)):
        col = names[i] + "_ret"
        a.append(col)
        data[col] = np.log(data[names[i]] / data[names[i]].shift(1))
    data.dropna(inplace=True)
    for i in range(len(a)):
        col_2 = names[i] + "_cret"
        b.append(col_2)
        data[col_2] = data[a[i]].cumsum().apply(np.exp)
    return data


def get_pe_data(name, kind):

    if kind == "stock":
        adress = f"{DB}/industries/{watchlist[name]['indus']}/{name}/pe/pe.xlsx"
        pe = pd.read_excel(
            adress,
            engine="openpyxl",
            usecols="B,C,S,R,Q",
            skiprows=7,
            parse_dates=["تاریخ میلادی"],
        )
    elif kind == "index":
        adress = f"{DB}/industries/{name}/index.xlsx"
        pe = pd.read_excel(
            adress,
            engine="openpyxl",
            usecols="B,C,K,L,M",
            skiprows=7,
            parse_dates=["تاریخ میلادی"],
        )
    pe.rename(columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jdate"}, inplace=True)
    pe.set_index("Date", inplace=True)
    for i in pe.index:
        for j in pe.columns:
            if pe.loc[i, j] == "-":
                pe.loc[i, j] = np.nan
    pe.dropna(inplace=True)
    for i in np.linspace(1, 4, 20):
        prob = len(pe[pe["P/E-ttm"] < pe["P/E-ttm"].median() * i]) / len(pe)
        if prob > 0.8:
            norm = i
            break
    pe_norm = pe[pe["P/E-ttm"] < norm * pe["P/E-ttm"].median()]
    pe_u = pe[pe["P/E-ttm"] > norm * pe["P/E-ttm"].median()]

    return pe, pe_norm, pe_u

def select_df(df,str1,str2):
    first = []
    end=[]
    ########search for str1 str2##########
    for i in df.index:
        for j in df.columns:
            if df.loc[i,j]==str1:
                first.append(i)

            if df.loc[i,j]==str2:
                end.append(i)
    a=[]
    #########search for str2 subsequent str1##########
    for i in end:
        if i-first[0]>0:
            a.append(i)
 
    resault=df.loc[first[0]:a[0]]
    ############preprocess resault#############
    resault.dropna(axis=1,how='all',inplace=True)
    resault.dropna(axis=0,how='all',inplace=True)
    # remove '-' from data
    for i in resault.index:
        for j in resault.columns:
            if resault.loc[i, j] == "-":
                resault.loc[i, j] = 1         
    return resault    

def delete_empty(df):
    c=0
    empty=[]
    for i in df.columns:
        for j in df[i]:
            if (j==1)|(j==0):
                c+=1
        if c==len(df):
            empty.append(i)     
        c=0
    df.drop(empty,axis=1,inplace=True)      

class DesiredPortfolio:
    def __init__(self, names, farsi, start, end):
        self.names = names
        self.farsi = farsi
        self.start = start
        self.end = end
        self.data, self.col_cret, self.col_ret = desired_portfolio(
            self.names, self.farsi, self.start, self.end
        )
        self.risk = pd.DataFrame(columns=["Mean", "Risk"])
        self.risk["Mean"] = self.data[self.col_ret].mean()
        self.risk["Risk"] = self.data[self.col_ret].std()

    def plot_port(self):
        self.data[self.col_cret].plot(figsize=[20, 8], subplots=True)


class Market:
    def __init__(self, year_s, month_s, day_s, year_end, month_end, day_end):
        self.start_date = pd.to_datetime(
            JalaliDate(year_s, month_s, day_s).to_gregorian()
        )
        self.end_date = pd.to_datetime(
            JalaliDate(year_end, month_end, day_end).to_gregorian()
        )
        self.shkhes_kol = read_index("شاخص کل6", "d", self.start_date, self.end_date)
        self.shkhes_ham = read_index(
            "شاخص کل (هم وزن)6", "d", self.start_date, self.end_date
        )
        self.daro = read_index("43-مواد دارویی6", "d", self.start_date, self.end_date)
        self.palayeshgah = read_index(
            "23-فراورده نفتی6", "d", self.start_date, self.end_date
        )
        self.kani_felezi = read_index(
            "13-کانه فلزی6", "d", self.start_date, self.end_date
        )
        self.felezat_asasi = read_index(
            "27-فلزات اساسی6", "d", self.start_date, self.end_date
        )
        self.chemical = read_index("44-شیمیایی6", "d", self.start_date, self.end_date)
        self.khodro = read_index("34-خودرو6", "d", self.start_date, self.end_date)
        self.utility = read_index(
            "40-تامین آب،برق،گ6", "d", self.start_date, self.end_date
        )
        self.ghaza = read_index(
            "42-غذایی بجز قند6", "d", self.start_date, self.end_date
        )
        self.kashi = read_index(
            "49-کاشی و سرامیک6", "d", self.start_date, self.end_date
        )
        self.cement = read_index("53-سیمان6", "d", self.start_date, self.end_date)
        self.bank = read_index("57-بانکها6", "d", self.start_date, self.end_date)
        self.haml = read_index("60-حمل و نقل6", "d", self.start_date, self.end_date)
        self.ghand = read_index("38-قند و شکر6", "d", self.start_date, self.end_date)
        self.shkhes_kol_w = read_index("شاخص کل6", "w", self.start_date, self.end_date)
        self.shkhes_ham_w = read_index(
            "شاخص کل (هم وزن)6", "w", self.start_date, self.end_date
        )
        self.daro_w = read_index("43-مواد دارویی6", "w", self.start_date, self.end_date)
        self.palayeshgah_w = read_index(
            "23-فراورده نفتی6", "w", self.start_date, self.end_date
        )
        self.kani_felezi_w = read_index(
            "13-کانه فلزی6", "w", self.start_date, self.end_date
        )
        self.felezat_asasi_w = read_index(
            "27-فلزات اساسی6", "w", self.start_date, self.end_date
        )
        self.chemical_w = read_index("44-شیمیایی6", "w", self.start_date, self.end_date)
        self.khodro_w = read_index("34-خودرو6", "w", self.start_date, self.end_date)
        self.utility_w = read_index(
            "40-تامین آب،برق،گ6", "w", self.start_date, self.end_date
        )
        self.ghaza_w = read_index(
            "42-غذایی بجز قند6", "w", self.start_date, self.end_date
        )
        self.kashi_w = read_index(
            "49-کاشی و سرامیک6", "w", self.start_date, self.end_date
        )
        self.cement_w = read_index("53-سیمان6", "w", self.start_date, self.end_date)
        self.bank_w = read_index("57-بانکها6", "w", self.start_date, self.end_date)
        self.haml_w = read_index("60-حمل و نقل6", "w", self.start_date, self.end_date)
        self.ghand_w = read_index("38-قند و شکر6", "w", self.start_date, self.end_date)
        df = pd.DataFrame(columns=["Shakhes_Kol"])
        df["Shakhes_Kol"] = self.shkhes_kol["Close"]
        df["Shakhes_ham"] = self.shkhes_ham["Close"]
        df["Palayeshgah"] = self.palayeshgah["Close"]
        df["Felezat"] = self.felezat_asasi["Close"]
        df["Chemical"] = self.chemical["Close"]
        df["Kani_Felezi"] = self.kani_felezi["Close"]
        df["Utility"] = self.utility["Close"]
        df["Bank"] = self.bank["Close"]
        df["Daro"] = self.daro["Close"]
        df["Khodro"] = self.khodro["Close"]
        df["haml"] = self.haml["Close"]
        df["Ghand"] = self.ghand["Close"]
        df["Ghaza"] = self.ghaza["Close"]
        df["kashi"] = self.kashi["Close"]
        df["Cement"] = self.cement["Close"]
        self.df = df

    def plot_value(self):
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 2, 1)
        plt.hist(self.shkhes_kol["Value"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_kol["Value"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_kol["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_shkhes_kol")
        plt.subplot(2, 2, 2)
        plt.hist(self.shkhes_ham["Value"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_ham["Value"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_ham["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_shkhes_ham")
        plt.subplot(2, 2, 3)
        plt.hist(self.shkhes_kol_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_kol_w["Value"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_kol_w["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_shkhes_kol_Weekly")
        plt.subplot(2, 2, 4)
        plt.hist(self.shkhes_ham_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_ham_w["Value"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_ham_w["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_shkhes_ham_Weekly")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 3, 1)
        plt.hist(self.bank["Value"], edgecolor="black", bins=60)
        plt.axvline(self.bank["Value"].iloc[-1], color="red")
        plt.axvline(self.bank["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_Bank")
        plt.subplot(2, 3, 2)
        plt.hist(self.khodro["Value"], edgecolor="black", bins=60)
        plt.axvline(self.khodro["Value"].iloc[-1], color="red")
        plt.axvline(self.khodro["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_khodro")
        plt.subplot(2, 3, 3)
        plt.hist(self.haml["Value"], edgecolor="black", bins=60)
        plt.axvline(self.haml["Value"].iloc[-1], color="red")
        plt.axvline(self.haml["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_haml")
        plt.subplot(2, 3, 4)
        plt.hist(self.bank_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.bank_w["Value"].iloc[-1], color="red")
        plt.axvline(self.bank_w["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_Bank_Weekly")
        plt.subplot(2, 3, 5)
        plt.hist(self.khodro_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.khodro_w["Value"].iloc[-1], color="red")
        plt.axvline(self.khodro_w["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_khodro_Weekly")
        plt.subplot(2, 3, 6)
        plt.hist(self.haml_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.haml_w["Value"].iloc[-1], color="red")
        plt.axvline(self.haml_w["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_haml_Weekly")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 3, 1)
        plt.hist(self.felezat_asasi["Value"], edgecolor="black", bins=60)
        plt.axvline(self.felezat_asasi["Value"].iloc[-1], color="red")
        plt.axvline(
            self.felezat_asasi["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_felezat_asasi")
        plt.subplot(2, 3, 2)
        plt.hist(self.chemical["Value"], edgecolor="black", bins=60)
        plt.axvline(self.chemical["Value"].iloc[-1], color="red")
        plt.axvline(self.chemical["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_chemical")
        plt.subplot(2, 3, 3)
        plt.hist(self.palayeshgah["Value"], edgecolor="black", bins=60)
        plt.axvline(self.palayeshgah["Value"].iloc[-1], color="red")
        plt.axvline(
            self.palayeshgah["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_palayeshgah")
        plt.subplot(2, 3, 4)
        plt.hist(self.felezat_asasi_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.felezat_asasi_w["Value"].iloc[-1], color="red")
        plt.axvline(
            self.felezat_asasi_w["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_felezat_asasi_Weekly")
        plt.subplot(2, 3, 5)
        plt.hist(self.chemical_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.chemical_w["Value"].iloc[-1], color="red")
        plt.axvline(
            self.chemical_w["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_chemical_Weekly")
        plt.subplot(2, 3, 6)
        plt.hist(self.palayeshgah_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.palayeshgah_w["Value"].iloc[-1], color="red")
        plt.axvline(
            self.palayeshgah_w["Value"].median(), linestyle="dashed", color="black"
        )
        plt.title("Voloume_Of_palayeshgah_Weekly")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 3, 1)
        plt.hist(self.daro["Value"], edgecolor="black", bins=60)
        plt.axvline(self.daro["Value"].iloc[-1], color="red")
        plt.axvline(self.daro["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_daro")
        plt.subplot(2, 3, 2)
        plt.hist(self.ghand["Value"], edgecolor="black", bins=60)
        plt.axvline(self.ghand["Value"].iloc[-1], color="red")
        plt.axvline(self.ghand["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_ghand")
        plt.subplot(2, 3, 3)
        plt.hist(self.ghaza["Value"], edgecolor="black", bins=60)
        plt.axvline(self.ghaza["Value"].iloc[-1], color="red")
        plt.axvline(self.ghaza["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_ghaza")
        plt.subplot(2, 3, 4)
        plt.hist(self.daro_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.daro_w["Value"].iloc[-1], color="red")
        plt.axvline(self.daro_w["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_daro_Weekly")
        plt.subplot(2, 3, 5)
        plt.hist(self.ghand_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.ghand_w["Value"].iloc[-1], color="red")
        plt.axvline(self.ghand_w["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_ghand_Weekly")
        plt.subplot(2, 3, 6)
        plt.hist(self.ghaza_w["Value"], edgecolor="black", bins=60)
        plt.axvline(self.ghaza_w["Value"].iloc[-1], color="red")
        plt.axvline(self.ghaza_w["Value"].median(), linestyle="dashed", color="black")
        plt.title("Voloume_Of_ghaza_Weekly")

    def plot_ret(self):
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 2, 1)
        plt.hist(self.shkhes_kol["Change"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_kol["Change"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_kol["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_shkhes_kol")
        plt.subplot(2, 2, 2)
        plt.hist(self.shkhes_ham["Change"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_ham["Change"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_ham["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_shkhes_ham")
        plt.subplot(2, 2, 3)
        plt.hist(self.shkhes_kol_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_kol_w["Change"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_kol_w["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_shkhes_kol_Weekly")
        plt.subplot(2, 2, 4)
        plt.hist(self.shkhes_ham_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.shkhes_ham_w["Change"].iloc[-1], color="red")
        plt.axvline(
            self.shkhes_ham_w["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_shkhes_ham_Weekly")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 3, 1)
        plt.hist(self.bank["Change"], edgecolor="black", bins=60)
        plt.axvline(self.bank["Change"].iloc[-1], color="red")
        plt.axvline(self.bank["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_Bank")
        plt.subplot(2, 3, 2)
        plt.hist(self.khodro["Change"], edgecolor="black", bins=60)
        plt.axvline(self.khodro["Change"].iloc[-1], color="red")
        plt.axvline(self.khodro["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_khodro")
        plt.subplot(2, 3, 3)
        plt.hist(self.haml["Change"], edgecolor="black", bins=60)
        plt.axvline(self.haml["Change"].iloc[-1], color="red")
        plt.axvline(self.haml["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_haml")
        plt.subplot(2, 3, 4)
        plt.hist(self.bank_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.bank_w["Change"].iloc[-1], color="red")
        plt.axvline(self.bank_w["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_Bank_Weekly")
        plt.subplot(2, 3, 5)
        plt.hist(self.khodro_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.khodro_w["Change"].iloc[-1], color="red")
        plt.axvline(self.khodro_w["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_khodro_Weekly")
        plt.subplot(2, 3, 6)
        plt.hist(self.haml_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.haml_w["Change"].iloc[-1], color="red")
        plt.axvline(self.haml_w["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_haml_Weekly")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 3, 1)
        plt.hist(self.felezat_asasi["Change"], edgecolor="black", bins=60)
        plt.axvline(self.felezat_asasi["Change"].iloc[-1], color="red")
        plt.axvline(
            self.felezat_asasi["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_felezat_asasi")
        plt.subplot(2, 3, 2)
        plt.hist(self.chemical["Change"], edgecolor="black", bins=60)
        plt.axvline(self.chemical["Change"].iloc[-1], color="red")
        plt.axvline(self.chemical["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_chemical")
        plt.subplot(2, 3, 3)
        plt.hist(self.palayeshgah["Change"], edgecolor="black", bins=60)
        plt.axvline(self.palayeshgah["Change"].iloc[-1], color="red")
        plt.axvline(
            self.palayeshgah["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_palayeshgah")
        plt.subplot(2, 3, 4)
        plt.hist(self.felezat_asasi_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.felezat_asasi_w["Change"].iloc[-1], color="red")
        plt.axvline(
            self.felezat_asasi_w["Change"].mean(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_felezat_asasi_Weekly")
        plt.subplot(2, 3, 5)
        plt.hist(self.chemical_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.chemical_w["Change"].iloc[-1], color="red")
        plt.axvline(
            self.chemical_w["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_chemical_Weekly")
        plt.subplot(2, 3, 6)
        plt.hist(self.palayeshgah_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.palayeshgah_w["Change"].iloc[-1], color="red")
        plt.axvline(
            self.palayeshgah_w["Change"].median(), linestyle="dashed", color="black"
        )
        plt.title("Return_Of_palayeshgah_Weekly")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 3, 1)
        plt.hist(self.daro["Change"], edgecolor="black", bins=60)
        plt.axvline(self.daro["Change"].iloc[-1], color="red")
        plt.axvline(self.daro["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_daro")
        plt.subplot(2, 3, 2)
        plt.hist(self.ghand["Change"], edgecolor="black", bins=60)
        plt.axvline(self.ghand["Change"].iloc[-1], color="red")
        plt.axvline(self.ghand["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_ghand")
        plt.subplot(2, 3, 3)
        plt.hist(self.ghaza["Change"], edgecolor="black", bins=60)
        plt.axvline(self.ghaza["Change"].iloc[-1], color="red")
        plt.axvline(self.ghaza["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_ghaza")
        plt.subplot(2, 3, 4)
        plt.hist(self.daro_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.daro_w["Change"].iloc[-1], color="red")
        plt.axvline(self.daro_w["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_daro_Weekly")
        plt.subplot(2, 3, 5)
        plt.hist(self.ghand_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.ghand_w["Change"].iloc[-1], color="red")
        plt.axvline(self.ghand_w["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_ghand_Weekly")
        plt.subplot(2, 3, 6)
        plt.hist(self.ghaza_w["Change"], edgecolor="black", bins=60)
        plt.axvline(self.ghaza_w["Change"].iloc[-1], color="red")
        plt.axvline(self.ghaza_w["Change"].median(), linestyle="dashed", color="black")
        plt.title("Return_Of_ghaza_Weekly")

    def plot_indices(self):
        self.df[["Shakhes_ham", "Shakhes_Kol"]].plot(
            secondary_y="Shakhes_Kol", figsize=[20, 12]
        )
        self.df[["Palayeshgah", "Shakhes_Kol"]].plot(
            secondary_y="Shakhes_Kol", figsize=[20, 12]
        )
        self.df[["Felezat", "Shakhes_Kol"]].plot(
            secondary_y="Shakhes_Kol", figsize=[20, 12]
        )
        self.df[["Kani_Felezi", "Shakhes_Kol"]].plot(
            secondary_y="Shakhes_Kol", figsize=[20, 12]
        )
        self.df[["Chemical", "Shakhes_Kol"]].plot(
            secondary_y="Shakhes_Kol", figsize=[20, 12]
        )
        self.df[["Utility", "Shakhes_Kol"]].plot(
            secondary_y="Shakhes_Kol", figsize=[20, 12]
        )
        self.df[["Bank", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["Daro", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["Khodro", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["haml", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["Ghand", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["Ghaza", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["kashi", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["Cement", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )
        self.df[["Shakhes_Kol", "Shakhes_ham"]].plot(
            secondary_y="Shakhes_ham", figsize=[20, 12]
        )


class Portfolio:
    def __init__(self, broker, owner, alpha, date):
        self.broker = broker
        self.owner = owner
        self.alpha = alpha
        self.date = date
        self.Port = read_portfolio(self.broker, self.owner, self.date, self.alpha)
        self.Group = group_portFolio(self.Port)
        self.Data, self.History = history(self.broker, self.owner)

    def plot_group(self):
        plt.figure(figsize=[20, 8])
        plt.pie(
            self.Group.iloc[0:-1]["Value"],
            labels=self.Group.index[0:-1],
            autopct="%1.1f%%",
            textprops={"color": "w"},
            shadow=True,
            explode=(0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

    def plot_history(self):
        plt.figure(figsize=[15, 8])
        plt.plot(self.History["Value"], marker="o", linewidth=3)


class Macro:
    def __init__(
        self,
        year_s=1391,
        month_s=1,
        year_end=1401,
        month_end=12,
        pe_f=6.5,
        sma_s=20,
        sma_l=100,
    ):
        self.start = pd.to_datetime(JalaliDate(year_s, month_s, 1).to_gregorian())
        self.end = pd.to_datetime(JalaliDate(year_end, month_end, 1).to_gregorian())
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.dollar_azad, self.dollar_nima = read_dollar(self.start, self.end)
        self.dollar_m = self.dollar_azad.resample("M").last()
        self.dollar_m["Ret"] = self.dollar_azad.resample("M").sum()["Ret"]
        self.dollar_m["Cret"] = self.dollar_m["Ret"].cumsum().apply(np.exp)
        self.dollar_m["Change"] = (
            self.dollar_m["Cret"] / self.dollar_m["Cret"].shift(1)
        ) - 1
        self.dollar_y = self.dollar_azad.resample("Y").last()
        self.dollar_y["Ret"] = self.dollar_azad.resample("Y").sum()["Ret"]
        self.dollar_y["Cret"] = self.dollar_y["Ret"].cumsum().apply(np.exp)
        self.dollar_y["Change"] = (
            self.dollar_y["Cret"] / self.dollar_y["Cret"].shift(1)
        ) - 1
        self.shakhes_kol = read_index("شاخص کل6", "d", self.start, self.end)
        self.shkhes_ham = read_index("شاخص کل (هم وزن)6", "d", self.start, self.end)
        self.sma = 100
        self.dev = 3
        self.pe, self.probably = read_pe(self.sma, self.dev, self.start, self.end)
        self.direct = read_direct(self.start, self.end, self.sma_s, self.sma_l)
        # self.industry = read_industry()
        self.IR = read_ir()
        self.summary = pd.DataFrame(columns=["PE"])
        self.summary["IR"] = self.IR["Close"]
        self.summary["PE"] = self.pe["Close"]
        self.summary["dollar"] = self.dollar_azad["Close"]
        self.pe_f = pe_f
        self.value = read_value(self.start, self.end, sma_s, sma_l)
        self.summary.dropna(inplace=True)
        model = linear.LinearRegression()
        model.fit(self.summary[["IR"]], self.summary["PE"])
        pred = model.predict(self.summary[["IR"]])
        self.summary["Pred"] = pred
        # create monetary data
        self.get_historical_data()

    def plot_dollar(self):
        plt.figure(figsize=[20, 12])
        plt.subplot(2, 3, 1)
        plt.hist(self.dollar_azad["Ret"], bins=60, edgecolor="black")
        plt.axvline(self.dollar_azad["Ret"].mean(), linestyle="dashed", color="black")
        plt.axvline(self.dollar_azad["Ret"].iloc[-1], color="red")
        plt.title("Daily_Hist")
        plt.subplot(2, 3, 2)
        plt.hist(self.dollar_m["Ret"], bins=60, edgecolor="black")
        plt.axvline(self.dollar_m["Ret"].mean(), linestyle="dashed", color="black")
        plt.axvline(self.dollar_m["Ret"].iloc[-1], color="red")
        plt.title("Monthly_Hist")
        plt.subplot(2, 3, 3)
        plt.hist(self.dollar_y["Ret"], bins=60, edgecolor="black")
        plt.axvline(self.dollar_y["Ret"].mean(), linestyle="dashed", color="black")
        plt.axvline(self.dollar_y["Ret"].iloc[-1], color="red")
        plt.title("Yearly_Hist")
        plt.subplot(2, 3, 4)
        plt.plot(self.dollar_azad["Close"])
        plt.subplot(2, 3, 5)
        plt.plot(self.dollar_m["Close"], marker="o")
        plt.subplot(2, 3, 6)
        plt.plot(self.dollar_y["Close"], marker="o")
        plt.figure(figsize=[20, 8])
        plt.plot(self.dollar_azad["Close"], label="azad")
        plt.plot(self.dollar_nima["Close"], label="nima")
        plt.legend(fontsize=15)

    def plot_pe(self):
        pe_mean = self.pe["Close"].median()
        pe_std = self.pe["Close"].std()
        pe1 = pe_mean + pe_std
        pe2 = pe_mean + 2 * pe_std
        pe3 = pe_mean + 3 * pe_std
        plt.figure(figsize=[15, 8])
        plt.hist(self.pe["Close"], edgecolor="black", bins=100)
        plt.axvline(pe_mean, color="black", linestyle="dashed")
        plt.axvline(pe1, color="green", linestyle="dashed")
        plt.axvline(pe2, color="blue", linestyle="dashed")
        plt.axvline(pe3, color="red", linestyle="dashed")
        plt.axvline(self.pe["Close"].iloc[-1], color="red")
        plt.axvline(5, color="gray")
        plt.axvline(self.pe_f, color="blue")
        plt.title("P/E Historical")
        self.pe["SMA"] = self.pe["Close"].rolling(self.sma).mean()
        self.pe["Upper"] = (
            self.pe["SMA"] + self.dev * self.pe["Close"].rolling(20).std()
        )
        self.pe["Lower"] = (
            self.pe["SMA"] - self.dev * self.pe["Close"].rolling(20).std()
        )
        self.pe[["Upper", "Close", "Lower", "SMA"]].plot(figsize=[20, 8])
        plt.figure(figsize=[20, 8])
        plt.plot(self.shakhes_kol["Cret"], label="Shakhes")
        plt.plot(self.dollar_azad["Cret"], label="Dollar")
        plt.legend()
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.summary["IR"], self.summary["PE"], "b.")
        plt.plot(self.summary["IR"], self.summary["Pred"])
        plt.title("PE & IR correaltion")
        plt.subplot(1, 2, 2)
        plt.plot(self.summary["PE"], self.summary["Pred"], "b.")
        plt.plot(self.summary["PE"], self.summary["PE"])

    def plot_direct(self):
        direct_mean = self.direct["Pure"].mean()
        direct_std = self.direct["Pure"].std()
        direct_w = self.direct.resample("W").mean()
        direct_M = self.direct.resample("M").mean()
        direct1 = direct_mean + direct_std
        direct_1 = direct_mean - direct_std
        direct2 = direct_mean + 2 * direct_std
        direct_2 = direct_mean - 2 * direct_std
        direct3 = direct_mean + 3 * direct_std
        direct_3 = direct_mean - 3 * direct_std
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 3, 1)
        plt.hist(self.direct["Pure"], edgecolor="black", bins=150)
        plt.axvline(direct_mean, color="black", linestyle="dashed")
        plt.axvline(direct1, color="green", linestyle="dashed")
        plt.axvline(direct_1, color="green", linestyle="dashed")
        plt.axvline(direct2, color="blue", linestyle="dashed")
        plt.axvline(direct_2, color="blue", linestyle="dashed")
        plt.axvline(direct3, color="red", linestyle="dashed")
        plt.axvline(direct_3, color="red", linestyle="dashed")
        plt.axvline(self.direct["Pure"].iloc[-1], color="red")
        plt.title("Money_Direct_Daily")
        plt.subplot(1, 3, 2)
        plt.hist(direct_w["Pure"], edgecolor="black", bins=150)
        plt.axvline(direct_w.iloc[-1]["Pure"], color="red")
        plt.axvline(direct_w["Pure"].mean(), color="black", linestyle="dashed")
        plt.title("Money_Direct_Weekly")
        plt.subplot(1, 3, 3)
        plt.hist(direct_M["Pure"], edgecolor="black", bins=150)
        plt.axvline(direct_M.iloc[-1]["Pure"], color="red")
        plt.axvline(direct_M["Pure"].mean(), color="black", linestyle="dashed")
        plt.title("Money_Direct_Monthly")
        plt.figure(figsize=[20, 12])
        plt.subplot(3, 1, 1)
        plt.plot(direct_w["Pure"], marker="o")
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.title("Weekly_Direct")
        plt.subplot(3, 1, 2)
        plt.plot(direct_M["Pure"], marker="o")
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.title("Monthly_Direct")
        plt.subplot(3, 1, 3)
        plt.plot(self.direct["Pure"])
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.title("Daily_Direct")
        self.direct[["Pure", "Sma_s", "Sma_l"]].plot(figsize=[20, 8])
        plt.axhline(y=0, linestyle="dashed")

    def plot_industry(self, data_1, data_2):
        name = self.industry.index
        plt.figure(figsize=[20, 8])
        plt.scatter(self.industry[data_1], self.industry[data_2], s=15)
        if data_2 == "Return 1D":
            a = 0.05
        else:
            a = 0.2
        plt.axhline(0, color="black", linestyle="dashed")
        # plt.axhline(a,color='red',linestyle='dashed')
        # plt.axhline(-a,color='red',linestyle='dashed')
        plt.xlabel(data_1)
        plt.ylabel(data_2)
        for i in range(len(name)):
            plt.annotate(
                name[i],
                xy=(
                    self.industry[data_1].iloc[i] + 0.002,
                    self.industry[data_2].iloc[i] + 0.002,
                ),
            )

    def plot_pe_opt(self, opt, y_s, m_s, y_e, m_e):
        """IR dollar direct"""
        start = pd.to_datetime(JalaliDate(y_s, m_s, 1).to_gregorian())
        end = pd.to_datetime(JalaliDate(y_e, m_e, 1).to_gregorian())
        if opt == "IR":
            fig, ax = plt.subplots(figsize=[20, 8])
            ax2 = ax.twinx()
            ax.plot(self.pe["Close"][start:end], color="black", label="P/E-ttm")
            ax2.plot(self.IR["Close"][start:end], color="red", label="Interest_rate")
            ax2.legend()
        if opt == "dollar":
            fig, ax = plt.subplots(figsize=[20, 8])
            ax2 = ax.twinx()
            ax.plot(self.pe["Close"][start:end], color="black", label="P/E-ttm")
            ax2.plot(self.dollar_azad["Close"][start:end], color="red", label="dollar")
            ax2.legend()
        if opt == "direct":
            fig, ax = plt.subplots(figsize=[20, 8])
            ax2 = ax.twinx()
            ax.plot(self.pe["Close"][start:end], color="black", label="P/E-ttm")
            ax2.plot(self.direct["Pure"][start:end], color="red", label="direct")
            ax2.legend()

    def get_historical_data(self):
        # call all historical data
        history = pd.read_excel(f"{DB}/macro/my.xlsx")
        my_col = [
            "year",
            "dollar",
            "base_money",
            "paper_money",
            "current_deposits",
            "non_current_deposits",
            "cash",
            "oil_export",
            "non_oil_export",
            "total_export",
            "import",
            "constant_gdp",
            "current_gdp",
            "cpi_rate",
            "cpi",
            "ppi",
            "ppi_rate",
            "land",
            "land_rate",
            "dollar_land",
            "stock",
            "stock_rate",
        ]
        # rename coloumns
        for i in range(len(history.columns)):
            history.rename(columns={history.columns[i]: my_col[i]}, inplace=True)
        history.set_index("year", inplace=True)
        # add future data
        history.loc[1401,'dollar']=self.dollar_azad.iloc[-1]['Close']
        # extract monetary index
        moneatry = history[
            [
                "base_money",
                "paper_money",
                "current_deposits",
                "non_current_deposits",
                "cash",
            ]
        ]
        # extract price index
        price = history[
            ["dollar", "cpi", "ppi", "land", "dollar_land", "stock"]
        ]
        # extract foreign exchange data
        exchange = history[
            [
                "dollar",
                "oil_export",
                "non_oil_export",
                'total_export',
                "import",
                "current_gdp",
                "constant_gdp",
                "cpi",
                "cash",
            ]
        ]
        # create return of data
        exchange_ret = pd.DataFrame(columns=["dollar", "cpi"])
        exchange_ret["dollar"] = exchange["dollar"].pct_change()
        exchange_ret["gdp"] = exchange["constant_gdp"].pct_change()
        exchange_ret["cpi"] = exchange["cpi"].pct_change()
        exchange_ret["cash"] = exchange["cash"].pct_change()
        exchange_ret["oil_export"] = exchange["oil_export"].pct_change()
        exchange_ret["total_export"] = exchange["total_export"].pct_change()
        price_ret = pd.DataFrame(columns=["ppi", "cpi"])
        price_ret["cpi"] = price["cpi"].pct_change()
        price_ret["ppi"] = price["ppi"].pct_change()
        price_ret["land"] = price["land"].pct_change()
        price_ret["dollar"] = price["dollar"].pct_change()
        price_ret["stock"] = price["stock"].pct_change()
        price_ret.dropna(inplace=True)
        exchange_ret.dropna(inplace=True)
        # send data to self
        self.history = history
        self.monetary = moneatry
        self.price = price
        self.exchange = exchange
        self.exchange_ret = exchange_ret
        self.price_ret = price_ret


class Stock:
    def __init__(
        self,
        Name,
        year_s=1395,
        month_s=1,
        year_end=1401,
        month_end=12,
        year_tester_s=1400,
        year_tester_end=1401,
        month_tester_s=1,
        month_tester_end=12,
        n=5,
    ):

        self.Name = Name
        self.industry = watchlist[Name]["indus"]
        self.farsi = watchlist[Name]["token"]

        self.start_date = pd.to_datetime(JalaliDate(year_s, month_s, 1).to_gregorian())

        self.end_date = pd.to_datetime(
            JalaliDate(year_end, month_end, 1).to_gregorian()
        )

        self.n = n
        self.tester_start = pd.to_datetime(
            JalaliDate(year_tester_s, month_tester_s, 1).to_gregorian()
        )
        self.tester_end = pd.to_datetime(
            JalaliDate(year_tester_end, month_tester_end, 1).to_gregorian()
        )
        self.tc = 0.012
        ######## load price data ##########
        try:
            self.Price, self.Price_dollar = read_stock(
                self.farsi, self.start_date, self.end_date
            )
        except:
            print(f"cant download {self.Name} price")
        ######## load income yearly ############
        (
            self.income_rial_yearly,
            self.income_common_rial_yearly,
            self.Risk_income_yearly,
            self.cagr_rial_yearly,
            self.fiscal_year,
            self.last_year,
        ) = get_income_yearly(self.Name, "rial", self.n)
        (
            self.income_dollar_yearly,
            self.income_common_dollar_yearly,
            self.Risk_income_yearly,
            self.cagr_dollar_yearly,
            i,
            j,
        ) = get_income_yearly(self.Name, "dollar", self.n)
        (
            self.income_rial_quarterly,
            self.income_common_rial_quarterly,
            self.Risk_income_rial_quarterly,
            self.cagr_rial_quarterly,
        ) = get_income_quarterly(self.Name, "rial", self.n, self.fiscal_year)
        (
            self.income_dollar_quarterly,
            self.income_common_dollar_quarterly,
            self.Risk_income_dollar_quarterly,
            self.cagr_dollar_quarterly,
        ) = get_income_quarterly(self.Name, "dollar", self.n, self.fiscal_year)
        self.Buy_Power = type_record(self.farsi)
        self.Buy_Power_w = self.Buy_Power.resample("W").mean()
        self.Buy_Power_m = self.Buy_Power.resample("M").mean()
        self.dollar_analyse = (
            self.income_dollar_yearly[["Total_Revenue"]]
            / self.income_dollar_yearly.iloc[0]["Total_Revenue"]
        )
        ######### Load balancesheet ############
        try:
            self.get_balance_sheet("yearly")
            self.get_balance_sheet("quarterly")
        except:
            print(f"add balancs sheet {self.Name}")
        ########## Load cash_flow ##############
        try:
            self.get_cash_flow("yearly")
            self.get_cash_flow("quarterly")
        except:
            print(f"add cash_flow {self.Name}")

        self.dollar_analyse["Net_Profit"] = (
            self.income_dollar_yearly[["Net_Profit"]]
            / self.income_dollar_yearly.iloc[0]["Net_Profit"]
        )
        self.dollar_analyse["Total_change"] = self.income_dollar_yearly[
            ["Total_Revenue"]
        ].pct_change()
        self.dollar_analyse["Net_change"] = self.income_dollar_yearly[
            ["Net_Profit"]
        ].pct_change()
        self.hazine_yearly = self.income_common_rial_yearly[
            [
                "Cost_of_Revenue",
                "Operating_Expense",
                "Interest_Expense",
                "Tax_Provision",
            ]
        ]
        self.hazine_yearly = abs(self.hazine_yearly)
        # self.daramad_yearly=self.income_common_rial_yearly[['Other_operating_Income_Expense','Other_non_operating_Income_Expense']]
        self.hazine_quarterly = self.income_common_rial_quarterly[
            [
                "Cost_of_Revenue",
                "Operating_Expense",
                "Interest_Expense",
                "Tax_Provision",
            ]
        ]
        self.hazine_quarterly = abs(self.hazine_quarterly)
        # self.daramad_quarterly=self.income_common_rial_quarterly[['Other_operating_Income_Expense','Other_non_operate_Income_Expense']]
        self.Vp, self.Price_bin = voloume_profile(self.Price, "2020", 60)
        ############ create tester majule #############
        self.sma_tester = SmaTester(
            self.Price, self.tester_start, self.tester_end, self.tc
        )
        self.my_tester = TesterOneSide(
            self.Price, self.tester_start, self.tester_end, self.tc, self.Name
        )
        self.tester_price = TesterOneSidePrice(
            self.Price, self.tester_start, self.tester_end, self.tc
        )
        ############ Load_P/E Historical #############
        self.pe, self.pe_n, self.pe_u = get_pe_data(self.Name, "stock")
        self.dollar_azad, self.dollar_nima = read_dollar(self.start_date, self.end_date)
        mean_dollar = []
        mean_market = []
        ############ load product ###############
        try:
            self.get_product("yearly")
            self.get_product("monthly")
        except:
            print(f"add prouct data {self.Name}")

        ########### load cost ###########
        try:
            self.get_cost("yearly")
            self.get_cost("quarterly")
        except:
            print(f"add cost {self.Name}")
        ######### Load group p/e data #########
        try:
            self.group, self.gropu_n, self.group_u = get_pe_data(self.industry, "index")
        except:
            pass
        ########## Load  Optimize_Strategy file ############
        try:
            opt = pd.read_excel(f"{DB}/industries/{self.industry}/{self.Name}/opt.xlsx")
            # simulate with opt file
            self.my_tester.test_strategy(
                opt["SMA_s"].iloc[0],
                opt["SMA_l"].iloc[0],
                opt["VMA_S"].iloc[0],
                opt["VMA_l"].iloc[0],
            )
            self.opt = opt

        except:
            print("add opt file")
        ########### Predict future ############
        try:
            self.create_interest_data()
            self.predict_income()
            self.predict_balance_sheet()
            self.predict_interst()
            self.create_fcfe()
        except:
            print("cant predict future")
        ########### add risk of falling ############
        try:
            self.risk_falling = len(self.pe[self.pe["P/E-ttm"] < self.pe_fw]) / len(
                self.pe["P/E-ttm"]
            )
        except:
            print("cant calculate risk of falling")

        ######## add compare return data #########
        try:
            self.create_macro()
        except:
            print(f"cant compare returns {self.Name}")

    def plot_income_yearly(self):
        plt.figure(figsize=[20, 15])
        plt.subplot(3, 1, 1)
        plt.plot(self.income_rial_yearly["Total_Revenue"], marker="o")
        plt.title("Total_Revenue yearly")
        plt.subplot(3, 1, 2)
        plt.plot(self.income_rial_yearly["Net_Profit"], marker="o")
        plt.title("Net_Profit_yearlly")
        plt.subplot(3, 1, 3)
        plt.plot(self.income_rial_yearly["Gross_Profit"], marker="o")
        plt.title("Gross_Profit_yearlly")
        plt.figure(figsize=[20, 12])
        plt.subplot(2, 1, 1)
        plt.plot(self.income_common_rial_yearly["Gross_Profit"], marker="o")
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.title("Gross_Profit_margin")
        plt.subplot(2, 1, 2)
        plt.plot(self.income_common_rial_yearly["Net_Profit"], marker="o")
        plt.title("Net_Profit_margin")
        plt.figure(figsize=[20, 15])
        plt.subplot(3, 1, 1)
        plt.plot(self.income_dollar_yearly["Total_Revenue"], marker="o")
        plt.title("Total_Revenue yearly_dollar")
        plt.subplot(3, 1, 2)
        plt.plot(self.income_common_dollar_yearly["Net_Profit"], marker="o")
        plt.title("Net_Profit_yearlly_dollar")
        plt.subplot(3, 1, 3)
        plt.plot(self.income_dollar_yearly["Gross_Profit"], marker="o")
        plt.title("Gross_Profit_yearlly_dollar")

    def plot_income_quarterly(self):
        plt.figure(figsize=[20, 15])
        plt.subplot(3, 1, 1)
        plt.plot(self.income_rial_quarterly["Total_Revenue"], marker="o")
        plt.title("Total_Revenue quarterly")
        plt.subplot(3, 1, 2)
        plt.plot(self.income_rial_quarterly["Net_Profit"], marker="o")
        plt.title("Net_Profit_quarterly")
        plt.subplot(3, 1, 3)
        plt.plot(self.income_rial_quarterly["Gross_Profit"], marker="o")
        plt.title("Gross_Profit_quarterly")
        plt.figure(figsize=[20, 12])
        plt.subplot(2, 1, 1)
        plt.plot(self.income_common_rial_quarterly["Gross_Profit"], marker="o")
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.subplot(2, 1, 2)
        plt.hist(self.income_common_rial_quarterly["Gross_Profit"], edgecolor="black")
        plt.axvline(
            x=self.income_common_rial_quarterly["Gross_Profit"].iloc[-1], color="red"
        )
        plt.figure(figsize=[20, 15])
        plt.subplot(3, 1, 1)
        plt.plot(self.income_dollar_quarterly["Total_Revenue"], marker="o")
        plt.title("Total_Revenue quarterly")
        plt.subplot(3, 1, 2)
        plt.plot(self.income_dollar_quarterly["Net_Profit"], marker="o")
        plt.title("Net_Profit_quarterly")
        plt.subplot(3, 1, 3)
        plt.plot(self.income_dollar_quarterly["Gross_Profit"], marker="o")
        plt.title("Gross_Profit_quarterly")
        plt.figure(figsize=[20, 12])
        plt.subplot(2, 1, 1)
        plt.plot(self.income_common_dollar_quarterly["Gross_Profit"], marker="o")
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.subplot(2, 1, 2)
        plt.hist(self.income_common_dollar_quarterly["Gross_Profit"], edgecolor="black")
        plt.axvline(
            x=self.income_common_dollar_quarterly["Gross_Profit"].iloc[-1], color="red"
        )

    def plot_ret_hist(self, start, end):
        w_df = self.Price.resample("W").mean()
        m_df = self.Price.resample("M").mean()
        data = self.Price["Change"][start:end]
        data_w = w_df["Change"]
        data_m = m_df["Change"]
        plt.figure(figsize=[20, 12])
        plt.subplot(3, 1, 1)
        plt.hist(data, bins=60, edgecolor="black")
        plt.axvline(x=0, color="black")
        plt.axvline(x=0.05, color="black", linestyle="dashed")
        plt.axvline(x=-0.05, color="black", linestyle="dashed")
        plt.axvline(x=data.iloc[-1], color="red")
        plt.subplot(3, 1, 2)
        plt.hist(data_w, bins=60, edgecolor="black")
        plt.axvline(x=0, color="black")
        plt.axvline(x=0.05, color="black", linestyle="dashed")
        plt.axvline(x=-0.05, color="black", linestyle="dashed")
        plt.axvline(x=data_w.iloc[-1], color="red")
        plt.subplot(3, 1, 3)
        plt.hist(data_m, bins=60, edgecolor="black")
        plt.axvline(x=0, color="black")
        plt.axvline(x=0.05, color="black", linestyle="dashed")
        plt.axvline(x=-0.05, color="black", linestyle="dashed")
        plt.axvline(x=data_m.iloc[-1], color="red")

    def plot_buy_power(self, start, end):
        plt.figure(figsize=[20, 15])
        plt.subplot(3, 1, 1)
        plt.hist(self.Buy_Power[start:end], edgecolor="black", bins=60)
        plt.axvline(self.Buy_Power.iloc[-1]["Buy_Power"], color="red")
        plt.axvline(self.Buy_Power[start:end]["Buy_Power"].median(), linestyle="dashed")
        plt.title("His_Daily_Buy_Power")
        plt.subplot(3, 1, 2)
        plt.hist(self.Buy_Power_w[start:end], edgecolor="black", bins=60)
        plt.axvline(self.Buy_Power_w.iloc[-1]["Buy_Power"], color="red")
        plt.axvline(
            self.Buy_Power_w[start:end]["Buy_Power"].median(),
            color="black",
            linestyle="dashed",
        )
        plt.title("His_Weekly_Buy_Power")
        plt.subplot(3, 1, 3)
        plt.hist(self.Buy_Power_m[start:end], edgecolor="black", bins=60)
        plt.axvline(self.Buy_Power_m.iloc[-1]["Buy_Power"], color="red")
        plt.title("His_Monthly_Buy_Power")

    def plot_hazine(self):
        self.hazine_yearly.plot(marker="o", figsize=[20, 8])
        plt.title("Hazine_Yearly")
        self.hazine_quarterly.plot(marker="o", figsize=[20, 8])
        plt.title("Hazine_quarterly")

    def plot_voloume_profile(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10])
        ax1.barh(self.Price_bin, self.Vp, 100, edgecolor="black")
        ax1.grid(color="white")
        ax2.plot(self.Price["2020":]["Close"], linewidth=3)
        ax2.grid(color="white")

    def plot_pe(self):

        try:
            pe = self.pe_fw
        except:
            pe = self.pe["P/E-ttm"].iloc[0]
            print("No forward data available ")
        # float pe data
        data = []
        pe_data = self.pe["P/E-ttm"].values
        pe_2=self.Price['Close'].iloc[-1]/self.pred_income.loc[1402,'EPS_Capital']
        for i in pe_data:
            data.append(float(i))
        # fit f distribution to data
        f_param = stats.f.fit(data)
        f = stats.f(dfn=f_param[0], dfd=f_param[1], loc=f_param[2], scale=f_param[3])
        self.f = f
        x = np.linspace(f.ppf(0.01), f.ppf(0.99), 1000)
        ########   Histogeram Plot ##########
        plt.figure(figsize=[15, 10])
        plt.subplot(3, 1, 1)
        sns.distplot(self.pe["P/E-ttm"], kde="True")
        plt.plot(x, f.pdf(x))
        plt.axvline(self.pe["P/E-ttm"].median(), color="black", linestyle="dashed")
        plt.axvline(self.pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(pe, color="red", linestyle="dashed")
        plt.axvline(pe_2, color="red", linestyle="dashed",alpha=0.5)
        plt.title("all_pe_data")
        plt.subplot(3, 1, 2)
        plt.hist(self.pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(self.pe_n["P/E-ttm"].median(), color="black", linestyle="dashed")
        plt.axvline(self.pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(pe, color="red", linestyle="dashed")
        plt.axvline(pe_2, color="red", linestyle="dashed",alpha=0.5)
        plt.title("Normall_pe_data")
        plt.subplot(3, 1, 3)
        plt.hist(self.pe_u["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(self.pe_u["P/E-ttm"].median(), color="black", linestyle="dashed")
        plt.figure(figsize=[20, 10])
        ###########  Line_Plot  ############
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(self.pe["P/E-ttm"]["2015":], alpha=0.5, label="P/E")
        ax.legend()
        ax2.plot(self.Price["Close"]["2015":], color="black", label="Price")
        ax2.legend()
        ax.axhline(5, alpha=0.3)
        ax.axhline(self.pe_fw, linestyle="dashed", color="red")

        return pe

    def plot_resault(self):
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 3, 1)
        plt.plot(
            self.dollar_income["dollar_cret"], marker="o", color="black", label="dollar"
        )
        plt.plot(
            self.dollar_income["Market_cret"], marker="o", color="blue", label="Market"
        )
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(
            self.dollar_income["dollar_cret"], marker="o", color="black", label="dollar"
        )
        plt.plot(
            self.dollar_income["Net_Profit_cret"],
            marker="o",
            color="blue",
            label="Net_Profit",
        )
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(
            self.dollar_income["Market_cret"], marker="o", color="black", label="Market"
        )
        plt.plot(
            self.dollar_income["Net_Profit_cret"],
            marker="o",
            color="blue",
            label="Net_Profit",
        )
        plt.legend()
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 3, 1)
        plt.scatter(self.dollar_income["dollar"], self.dollar_income["Net_Profit"])
        plt.title("dollar and Net_Profit")
        plt.subplot(1, 3, 2)
        plt.scatter(self.dollar_income["dollar"], self.dollar_income["Net_margin"])
        plt.title("dollar and Net_margin")
        plt.subplot(1, 3, 3)
        plt.scatter(self.dollar_income["dollar"], self.dollar_income["Market"])
        plt.title("dollar and Market")

    def predict_price(self):
        df = self.eps_data.copy()
        # add future_year
        ratio_mean = df["ratio"][-2:].mean()
        future_year = df.index[-1] + 1
        df.loc[future_year] = np.zeros(len(df.iloc[0]))
        eps_future = self.pred_income.loc[future_year]["EPS"]
        df.loc[future_year, "EPS"] = eps_future
        # df.loc[future_year]['DPS']=df.loc[future_year]['EPS']*ratio_mean
        self.eps_data = df
        return df

    def predict_income(
        self,
        alpha_rate=1,
        alpha_prod=1,
        alpha_prod_next=1,
        alpha_rate_next=1,
        salary_g=1,
        material_g=1,
        energy_g=1,
        dep_g=1,
        transport_g=1,
        other_g=1,
        salary_g_next=1,
        material_g_next=1,
        energy_g_next=1,
        dep_g_next=1,
        transport_g_next=1,
        other_g_next=1,
    ):
        """
        alpha rate: predict last rate
        alpha_prod :predict product
        """
        self.alpha_rate = alpha_rate
        ###### estimate material alpha ######
        if ((self.industry == "ghaza") | (self.industry == "dode") |(self.industry=='darou') | (self.industry=='kashi')) & (material_g == 1):
            material_g = (
                np.mean(
                    self.my_cost_unit_quarterly["material"] / self.product_quarterly["Rate"]
                )
                * self.product_monthly["Rate"].iloc[-1]
                / self.my_cost_unit_quarterly["material"].iloc[-1]
            )
        self.material_g = material_g
        ######## call data ##########
        income_y = self.income_rial_yearly.copy()
        income_f = self.income_rial_quarterly.copy()
        income_common_f = self.income_common_rial_quarterly.copy()
        prod_y = self.product_yearly.copy()
        prod_m = self.product_monthly.copy()
        pred_cost_unit = self.my_cost_unit_quarterly.copy()
        interest = self.interest
        # weight_matrix
        w = [1, 3]
        # create future row
        future_year = income_y.index[-1] + 1
        income_y.loc[future_year] = np.zeros(len(income_y.iloc[0]))
        income_y.loc[future_year + 1] = np.zeros(len(income_y.iloc[0]))

        ########### calculate count of revenuee done and growth to last year ##########
        rev_done = 0
        count_rev_done = 0
        count_rev_last = 0
        rev_last = 0
        m = []
        # search in monthly product
        for i in prod_m.index:
            if (self.fiscal_year == 12) & (int(i.split("/")[0]) == future_year):
                rev_done += prod_m.loc[i]["Revenue"]
                count_rev_done += prod_m.loc[i]["Count"]
                m.append(i[5:7])
            if (self.fiscal_year != 12) & (
                (int(i.split("/")[0]) == future_year)
                | (
                    (int(i.split("/")[0]) == future_year - 1)
                    & (int(i.split("/")[1]) > self.fiscal_year)
                )
            ):
                rev_done += prod_m.loc[i]["Revenue"]
                count_rev_done += prod_m.loc[i]["Count"]
                # month done from fiscal_year
                m.append(i)

        self.m = m
        if self.fiscal_year == 12:
            last_index = ["1400" + "/" + f"{i}" for i in m]
        else:
            year = []
            month = []
            last_index = []
            for i in m:
                y = int(i.split("/")[0])
                mon = i.split("/")[1]
                month.append(mon)
                year.append(y)
            l = [i - 1 for i in year]
            for i in range(len(l)):
                s = f"{l[i]}" + "/" + f"{month[i]}"
                last_index.append(s)
        self.last_index = last_index
        for i in last_index:
            count_rev_last += prod_m.loc[i]["Count"]
            rev_last += prod_m.loc[i]["Revenue"]
        # Estimate growth of count sell
        growth_count_rev = count_rev_done / count_rev_last
        if alpha_prod == 1:
            alpha_prod = growth_count_rev
            self.alpha_prod = alpha_prod
        # average last 2 year and predict count rev
        count_rev_pred = alpha_prod * np.average(prod_y["Count"][-2:], weights=w)
        count_rev_residual = count_rev_pred - count_rev_done
        last_rate = 0
        last_rate = self.alpha_rate * prod_m.iloc[-1]["Rate"]
        self.last_rate = last_rate
        rev_residual = count_rev_residual * last_rate
        rev_pred = rev_residual + rev_done
        ############### Calculate cost of revenue ####################
        pred_cost_unit.loc[future_year] = pred_cost_unit[
            ["salary", "material", "energy", "depreciation", "transport", "other"]
        ].iloc[-1] * [salary_g, material_g, energy_g, dep_g, transport_g, other_g]
        pred_cost_unit.loc[future_year + 1] = pred_cost_unit[
            ["salary", "material", "energy", "depreciation", "transport", "other"]
        ].iloc[-1] * [
            salary_g_next,
            material_g_next,
            energy_g_next,
            dep_g_next,
            transport_g_next,
            other_g_next,
        ]
        # add total tp pred cost
        pred_cost_unit.loc[future_year, "total"] = pred_cost_unit.loc[future_year].sum()
        pred_cost_unit.loc[future_year, "profit"] = (
            last_rate - pred_cost_unit.loc[future_year]["total"]
        )
        pred_cost_unit.loc[future_year + 1, "total"] = pred_cost_unit.loc[
            future_year + 1
        ].sum()
        pred_cost_unit.loc[future_year + 1, "profit"] = (
            alpha_rate_next * last_rate - pred_cost_unit.loc[future_year + 1]["total"]
        )

        # add margin to pred cost
        temp_rate = self.product_quarterly["Rate"].values
        temp_rate = np.append(temp_rate, [last_rate, alpha_rate_next * last_rate])
        pred_cost_unit["margin"] = pred_cost_unit["profit"] / temp_rate
        self.pred_cost_unit = pred_cost_unit
        # creat pred product
        self.pred_prod = self.product_yearly.copy()
        self.pred_prod.loc[future_year] = np.zeros(len(self.pred_prod.iloc[0]))
        self.pred_prod.loc[future_year + 1] = np.zeros(len(self.pred_prod.iloc[0]))
        self.pred_prod.loc[future_year]["Count"] = count_rev_pred
        self.pred_prod.loc[future_year + 1]["Count"] = count_rev_pred * alpha_prod_next
        self.pred_prod.loc[future_year]["Rate"] = last_rate
        self.pred_prod.loc[future_year + 1]["Rate"] = last_rate * alpha_rate_next
        self.pred_prod.loc[future_year]["Revenue"] = (
            self.pred_prod.loc[future_year]["Count"]
            * self.pred_prod.loc[future_year]["Rate"]
        )
        self.pred_prod.loc[future_year + 1]["Revenue"] = (
            self.pred_prod.loc[future_year + 1]["Count"]
            * self.pred_prod.loc[future_year + 1]["Rate"]
        )
        # create_pred_cost_com
        rate_q = self.product_quarterly["Rate"].values
        rate_adj = np.append(rate_q, [last_rate, last_rate * alpha_rate_next])
        pred_cost_com = pd.DataFrame(columns=pred_cost_unit.columns)
        for i in pred_cost_com.columns:
            pred_cost_com[i] = pred_cost_unit[i] / rate_adj
        pred_cost_com.drop("margin", axis=1, inplace=True)
        self.pred_cost_com = pred_cost_com
        # add income to data frame
        income_y.loc[future_year]["Total_Revenue"] = int(rev_pred)
        income_y.loc[future_year + 1]["Total_Revenue"] = int(
            last_rate * count_rev_pred * alpha_prod_next * alpha_rate_next
        )
        ############# predict_cost_of_revenue ##################
        cost_rev_done = 0
        rev_cost_known = 0
        count_rev_cost_known = 0
        #search on quarterly income statement
        for i in income_f.index:
            if int(i[:4]) == future_year:
                cost_rev_done += (
                    income_f.loc[i]["Cost_of_Revenue"]
                    + income_f.loc[i]["Operating_Expense"]
                )
                rev_cost_known += income_f.loc[i]["Total_Revenue"]
        for i in self.product_quarterly.index:
            if int(i[:4]) == future_year:
                count_rev_cost_known += self.product_quarterly.loc[i]["Count"]
        count_rev_cost_unknown = count_rev_pred - count_rev_cost_known
        # predict_cost
        cost_pred = (
            cost_rev_done
            - count_rev_cost_unknown * pred_cost_unit.loc[future_year]["total"]
        )

        ############## predict operating expense ################
        op_exp_rev_ratio = (
            self.income_rial_yearly["Operating_Expense"]
            / self.income_rial_yearly["Total_Revenue"]
        )
        ratio_exp = np.average(op_exp_rev_ratio[-2:], weights=w)
        operating_expense_pred = ratio_exp * rev_pred
        # add operating expense to data frame
        income_y.loc[future_year]["Operating_Expense"] = int(operating_expense_pred)
        income_y.loc[future_year + 1]["Operating_Expense"] = int(
            ratio_exp * income_y.loc[future_year + 1]["Total_Revenue"]
        )
        # add cost to data frame
        income_y.loc[future_year]["Cost_of_Revenue"] = (
            cost_pred - operating_expense_pred
        )
        income_y.loc[future_year + 1]["Cost_of_Revenue"] = (
            -pred_cost_unit.loc[future_year + 1]["total"]
            * count_rev_pred
            * alpha_prod_next
            - income_y.loc[future_year + 1]["Operating_Expense"]
        )
        # add gross profit to data frame
        income_y.loc[future_year]["Gross_Profit"] = (
            income_y.loc[future_year]["Total_Revenue"]
            + income_y.loc[future_year]["Cost_of_Revenue"]
        )
        income_y.loc[future_year + 1]["Gross_Profit"] = (
            income_y.loc[future_year + 1]["Total_Revenue"]
            + income_y.loc[future_year + 1]["Cost_of_Revenue"]
        )
        ############ predict other operating ################
        other_rev_ratio = (
            self.income_rial_yearly["Other_operating_Income_Expense"]
            / self.income_rial_yearly["Total_Revenue"]
        )
        ratio_other = np.average(other_rev_ratio[-2:], weights=w)
        other_pred = ratio_other * rev_pred
        # add other operating to data frame
        income_y.loc[future_year]["Other_operating_Income_Expense"] = int(other_pred)
        income_y.loc[future_year + 1]["Other_operating_Income_Expense"] = int(
            ratio_other * income_y.loc[future_year + 1]["Total_Revenue"]
        )
        # add operating_income to data frame
        income_y.loc[future_year]["Operating_Income"] = (
            income_y.loc[future_year]["Gross_Profit"]
            + income_y.loc[future_year]["Operating_Expense"]
            + income_y.loc[future_year]["Other_operating_Income_Expense"]
        )
        income_y.loc[future_year + 1]["Operating_Income"] = (
            income_y.loc[future_year + 1]["Gross_Profit"]
            + income_y.loc[future_year + 1]["Operating_Expense"]
            + income_y.loc[future_year + 1]["Other_operating_Income_Expense"]
        )

        ############# add Other_non_operate_Income_Expense ################
        other_non_rev_ratio = (
            self.income_rial_yearly["Other_non_operate_Income_Expense"]
            / self.income_rial_yearly["Total_Revenue"]
        )
        ratio_other_non = np.average(other_non_rev_ratio[-2:], weights=w)
        ############# predic other non operating #################
        other_non_pred = ratio_other_non * rev_pred
        # add other non to dataframe
        income_y.loc[future_year]["Other_non_operate_Income_Expense"] = int(
            other_non_pred
        )
        income_y.loc[future_year + 1]["Other_non_operate_Income_Expense"] = int(
            ratio_other_non * income_y.loc[future_year + 1]["Total_Revenue"]
        )
        # add pre tax to data frame
        income_y.loc[future_year]["Pretax_Income"] = (
            income_y.loc[future_year]["Operating_Income"]
            + income_y.loc[future_year]["Interest_Expense"]
            + income_y.loc[future_year]["Other_non_operate_Income_Expense"]
        )
        income_y.loc[future_year + 1]["Pretax_Income"] = (
            income_y.loc[future_year + 1]["Operating_Income"]
            + income_y.loc[future_year + 1]["Interest_Expense"]
            + income_y.loc[future_year + 1]["Other_non_operate_Income_Expense"]
        )
        # add tax provision
        tax_ratio = (
            self.income_rial_yearly["Tax_Provision"]
            / self.income_rial_yearly["Pretax_Income"]
        )
        tax_ratio_mean = np.average(tax_ratio[-3:-1], weights=w)
        tax_pred = tax_ratio_mean * income_y.loc[future_year]["Pretax_Income"]
        income_y.loc[future_year]["Tax_Provision"] = int(tax_pred)
        income_y.loc[future_year]["Tax_Provision"] = int(
            tax_ratio_mean * income_y.loc[future_year]["Pretax_Income"]
        )
        # add net_income_common
        income_y.loc[future_year]["Net_Income_Common"] = (
            income_y.loc[future_year]["Pretax_Income"]
            + income_y.loc[future_year]["Tax_Provision"]
        )
        income_y.loc[future_year + 1]["Net_Income_Common"] = (
            income_y.loc[future_year + 1]["Pretax_Income"]
            + income_y.loc[future_year + 1]["Tax_Provision"]
        )
        income_y.loc[future_year]["Net_Profit"] = (
            income_y.loc[future_year]["Pretax_Income"]
            + income_y.loc[future_year]["Tax_Provision"]
        )
        income_y.loc[future_year + 1]["Net_Profit"] = (
            income_y.loc[future_year + 1]["Pretax_Income"]
            + income_y.loc[future_year + 1]["Tax_Provision"]
        )
        # add capital to data frame
        income_y.loc[future_year]["Capital"] = income_f["Capital"].iloc[-1]
        income_y.loc[future_year + 1]["Capital"] = income_f["Capital"].iloc[-1]
        # add eps to data frame
        income_y.loc[future_year]["EPS"] = (
            income_y.loc[future_year]["Net_Profit"]
            * 1000
            / income_y.loc[future_year]["Capital"]
        )
        income_y.loc[future_year + 1]["EPS"] = (
            income_y.loc[future_year + 1]["Net_Profit"]
            * 1000
            / income_y.loc[future_year + 1]["Capital"]
        )
        income_y.loc[future_year]["EPS_Capital"] = income_y.loc[future_year]["EPS"]
        income_y.loc[future_year + 1]["EPS_Capital"] = income_y.loc[future_year + 1][
            "EPS"
        ]
        ############## Create Hypothesis dictionary #############
        hypothesis = {
            "count_rev_pred": count_rev_pred,
            "count_rev_done": count_rev_done,
            "count_rev_last": count_rev_last,
            "growth count rev": growth_count_rev,
            "count_rev prod next": count_rev_pred * alpha_prod_next,
            "rev_done": rev_done,
            "rev_last": rev_last,
            "rev_residual": rev_residual,
            "total_rev": rev_pred,
            "Last_Rate": last_rate,
            "rate_next": last_rate * alpha_rate_next,
            "Cost_done": cost_rev_done,
            "Cost_pred": cost_pred,
            "Net_Profit": income_y.loc[future_year]["Net_Profit"],
        }
        parameters = {
            'alpha_prod_update':alpha_prod,
            'alpha_prod_next':alpha_prod_next,
            'alpha_rate_update':alpha_rate,
            'alpha_rate_net':alpha_rate_next,
            'material_g_update':material_g,
            'material_g_next':material_g_next,
            'salary_g_update':salary_g,
            'salary_g_next':salary_g_next,  
            'other_g_update':other_g,
            'other_g_next':other_g_next,
            'transport_g_update':transport_g,
            'transport_g_next':transport_g_next,
        }
        self.parameters=parameters
        self.pred_income = income_y
        self.hypothesis = hypothesis
        pred_com = pd.DataFrame(index=income_y.index, columns=income_y.columns)
        for i in income_y.index:
            pred_com.loc[i] = income_y.loc[i] / income_y.loc[i]["Total_Revenue"]
        self.pred_com = pred_com
        df = pd.concat([income_y.loc[[future_year - 1]], income_y.loc[[future_year]]])
        for i in df.index:
            for j in df.columns:
                if df.loc[i][j] == 0:
                    df.loc[i, j] = 0.1
        df = df / df.iloc[0]
        self.grow_income = df

    def plot_revenue(self):
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.product_yearly["Count"], marker="o")
        plt.title("Count_Of_Revenue")
        plt.subplot(1, 2, 2)
        plt.plot(self.product_yearly["Rate"], marker="o")
        plt.title("Rate")
        plt.figure(figsize=[20, 8])
        self.product_monthly[["Count", "cycle", "trend"]].plot(
            marker="o", figsize=[20, 8]
        )
        plt.title("Count_Of_Revenue_Monthly")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        self.product_yearly["Rate"].plot(marker="o")
        plt.title("yearly_rate")
        plt.subplot(1, 2, 2)
        self.product_monthly[
            -(self.fiscal_year - int(JalaliDate.today().isoformat().split("-")[1])) :
        ]["Rate"].plot(marker="o")
        plt.title("Quarterly rate")

    def create_eps_data(self):
        adress = f"{DB}/industries/{self.industry}/{self.Name}/eps.xlsx"
        df = pd.read_excel(adress, engine="openpyxl", index_col="year")
        df = df[::-1]
        # normalize_eps ans dps data
        for i in df.index:
            for j in df.columns:
                if df.loc[i][j] == "-":
                    df.loc[i, j] = np.nan
        df.dropna(inplace=True)
        df["capital_now"] = self.income_rial_quarterly.iloc[-1]["Capital"] * np.ones(
            [len(df["capital_now"]), 1]
        )
        df["EPS"] = df["EPS"] * df["capital"] / df["capital_now"]
        df["DPS"] = df["DPS"] * df["capital"] / df["capital_now"]
        df["ratio"] = df["DPS"] / df["EPS"]
        # add miladi
        lst = []
        for i in df["date"]:
            for j in i.split("/"):
                lst.append(int(j))
        l = int(len(lst) / 3)
        y = []
        m = []
        d = []
        miladi_lst = []
        for i in range(l):
            y = lst[3 * i]
            m = lst[3 * i + 1]
            d = lst[3 * i + 2]
            miladi = pd.to_datetime(JalaliDate(y, m, d).to_gregorian())
            miladi_lst.append(miladi)
        miladi_lst
        df["miladi"] = miladi_lst
        # add price_data
        price = []
        for i in df["miladi"]:
            price.append(self.Price.loc[i:].iloc[0]["Close"])
        df["price"] = price
        # add p/e and p/d
        df["pe"] = df["price"] / df["EPS"]
        df["pd"] = df["price"] / df["DPS"]

        ratio_mean = df["ratio"][-2:].mean()
        # add future_year
        future_year = df.index[-1] + 1
        df.loc[future_year] = np.zeros(len(df.iloc[0]))
        df.loc[future_year, "EPS"] = self.pred_income.loc[future_year]["EPS"]
        df.loc[future_year, "DPS"] = (
            self.pred_income.loc[future_year]["EPS"] * ratio_mean
        )
        df.loc[future_year, "capital"] = self.income_rial_quarterly.iloc[-1]["Capital"]
        df.loc[future_year, "capital_now"] = self.income_rial_quarterly.iloc[-1]["Capital"]
        df.loc[future_year, "ratio"] = (
            df.loc[future_year]["DPS"] / df.loc[future_year]["EPS"]
        )

        df.loc[future_year, "price"] = self.Price.iloc[-1]["Close"]
        df.loc[future_year, "pe"] = (
            df.loc[future_year]["price"] / df.loc[future_year]["EPS"]
        )
        df.loc[future_year, "pd"] = (
            df.loc[future_year]["price"] / df.loc[future_year]["DPS"]
        )
        # add future year + 1
        df.loc[future_year + 1] = np.zeros(len(df.iloc[0]))
        df.loc[future_year + 1, "EPS"] = self.pred_income.loc[future_year + 1]["EPS"]
        df.loc[future_year + 1, "DPS"] = (
            self.pred_income.loc[future_year + 1]["EPS"] * ratio_mean
        )
        df.loc[future_year + 1, "capital"] = self.income_rial_quarterly.iloc[-1]["Capital"]
        df.loc[future_year + 1, "capital_now"] = self.income_rial_quarterly.iloc[-1][
            "Capital"
        ]
        df.loc[future_year + 1, "ratio"] = (
            df.loc[future_year + 1]["DPS"] / df.loc[future_year + 1]["EPS"]
        )

        df.loc[future_year + 1, "price"] = self.Price.iloc[-1]["Close"]
        df.loc[future_year + 1, "pe"] = (
            df.loc[future_year + 1]["price"] / df.loc[future_year + 1]["EPS"]
        )
        df.loc[future_year + 1, "pd"] = (
            df.loc[future_year + 1]["price"] / df.loc[future_year + 1]["DPS"]
        )
        # add ret and cret
        df["EPS_ret"] = df["EPS"].pct_change()
        df["EPS_cret"] = df["EPS"] / df["EPS"].iloc[0]
        df["price_ret"] = df["price"].pct_change()
        df["price_cret"] = df["price"] / df["price"].iloc[0]
        self.eps_data = df
        self.grow_eps = self.eps_data["EPS_ret"].iloc[-1]
        self.pe_fw = df["pe"].loc[future_year]
        self.pe_med = self.pe_n["P/E-ttm"].median()
        self.potential_price_g = (self.pe_med - self.pe_fw) / self.pe_med

    def get_balance_sheet(self, periode):
        # create adress and coloumns proper tp peride
        if periode == "yearly":
            adress = (
                f"{DB}/industries/{self.industry}/{self.Name}/balancesheet/yearly.xlsx"
            )
            my_col = list(self.income_rial_yearly.index)
            my_col.insert(0, "Data")
        elif periode == "quarterly":
            adress = (
                f"{DB}/industries/{self.industry}/{self.Name}/balancesheet/quarterly.xlsx"
            )
            my_col = list(self.income_rial_quarterly.index)
            my_col.insert(0, "Data")
        # create raw data
        balance_original = pd.read_excel(adress)
        year = re.findall("[0123456789]{4}/[0123456789]{2}", str(balance_original))
        # select data
        balance = balance_original[
            (balance_original["Unnamed: 1"] == "دریافتنی‌های تجاری و سایر دریافتنی‌ها")
            | (balance_original["Unnamed: 1"] == "موجودی مواد و کالا")
            | (balance_original["Unnamed: 1"] == "پیش پرداخت ها")
            | (balance_original["Unnamed: 1"] == "موجودی نقد")
            | (balance_original["Unnamed: 1"] == "سرمایه گذاری کوتاه مدت")
            | (balance_original["Unnamed: 1"] == "داراییهای ثابت مشهود")
            | (balance_original["Unnamed: 1"] == "سرمایه گذاری در املاک")
            | (balance_original["Unnamed: 1"] == "سرمایه گذاریهای بلند مدت")
            | (
                balance_original["Unnamed: 1"]
                == "حسابها و اسناد دریافتنی تجاری بلند مدت"
            )
            | (
                balance_original["Unnamed: 1"]
                == "پرداختنی‌های تجاری و سایر پرداختنی‌ها"
            )
            | (balance_original["Unnamed: 1"] == "پیش دریافتها")
            | (balance_original["Unnamed: 1"] == "حصه جاری تسهیلات مالی دریافتی")
            | (balance_original["Unnamed: 1"] == "ذخیره مزایای پایان خدمت")
            | (balance_original["Unnamed: 1"] == "سود سهام پیشنهادی و پرداختنی")
            | (balance_original["Unnamed: 1"] == "سرمایه")
            | (balance_original["Unnamed: 1"] == "سود (زیان) انباشته")
            | (balance_original["Unnamed: 1"] == "داراییهای نامشهود")
        ]
        # my_col = year.insert(0, "Data")
        # my_col = ["Data", 1396, 1397, 1398, 1399, 1400]
        balance_sheet = balance.copy()
        balance_sheet.dropna(axis=1, inplace=True)
        # change balance_sheet coloumn
        for i in range(len(balance_sheet.columns)):
            balance_sheet.rename(
                columns={balance_sheet.columns[i]: my_col[i]}, inplace=True
            )
        #remove '-' from data
        for i in balance_sheet.index:
            for j in balance_sheet.columns:
                if balance_sheet.loc[i,j]=='-':
                    balance_sheet.loc[i,j]=1
        balance_sheet.set_index("Data", inplace=True)
        my_index = [
            "cash",
            "short term invest",
            "short term receivables",
            "inventory",
            "prepayment",
            "long term receivables",
            "long term invest",
            "invest in real estate",
            "tangible assets",
            "untangible assets",
            "payable",
            "pre received",
            "dividends payable",
            "financial facilities",
            "severance benefits",
            "capital",
            "retained earnings",
        ]
        # change balance sheet index
        for i in range(len(balance_sheet.index)):
            balance_sheet.rename(
                index={balance_sheet.index[i]: my_index[i]}, inplace=True
            )
        balance_sheet = balance_sheet.T
        balance_sheet["wc"] = (
            balance_sheet["short term receivables"]
            + balance_sheet["inventory"]
            + balance_sheet["prepayment"]
        ) - (balance_sheet["payable"] + balance_sheet["pre received"])
        # create common balance sheet to revenue
        balance_sheet_com = pd.DataFrame(
            index=balance_sheet.index, columns=balance_sheet.columns
        )
        # create differential_balance
        inv_balance = pd.DataFrame(
            columns=balance_sheet.columns, index=balance_sheet.index
        )
        # create differential_balance_com
        inv_balance_com = pd.DataFrame(
            columns=inv_balance.columns, index=inv_balance.index
        )
        # fill data_yearly
        if periode == "yearly":
            for i in balance_sheet_com.index:
                balance_sheet_com.loc[i] = (
                    balance_sheet.loc[i]
                    / self.income_rial_yearly.loc[i]["Total_Revenue"]
                )

            for i in balance_sheet.columns:
                inv_balance[i] = balance_sheet[i].diff(1)
            inv_balance.dropna(axis=0, inplace=True)

            for i in inv_balance.columns:
                inv_balance_com[i] = (
                    inv_balance[i] / self.income_rial_yearly["Total_Revenue"]
                )
        # fill data quarterly
        elif periode == "quarterly":
            for i in balance_sheet_com.index:
                balance_sheet_com.loc[i] = (
                    balance_sheet.loc[i]
                    / self.income_rial_quarterly.loc[i]["Total_Revenue"]
                )

            for i in balance_sheet.columns:
                inv_balance[i] = balance_sheet[i].diff(1)
            inv_balance.dropna(axis=0, inplace=True)

            for i in inv_balance.columns:
                inv_balance_com[i] = (
                    inv_balance[i] / self.income_rial_quarterly["Total_Revenue"]
                )
        # send to self
        if periode == "yearly":
            self.balance_sheet_yearly = balance_sheet
            self.balance_com_yearly = balance_sheet_com
            self.inv_balance_yearly = inv_balance
            self.inv_balance_com_yearly = inv_balance_com
        elif periode == "quarterly":
            self.balance_sheet_quarterly = balance_sheet
            self.balance_com_quarterly = balance_sheet_com
            self.inv_balance_quarterly = inv_balance
            self.inv_balance_com_quarterly = inv_balance_com

    def get_cash_flow(self, preiode):
        if preiode == "yearly":
            adress = f"{DB}/industries/{self.industry}/{self.Name}/cashflow/yearly.xlsx"
            my_col = list(self.income_rial_yearly.index)
            my_col.insert(0, "Data")
        elif preiode == "quarterly":
            adress = f"{DB}/industries/{self.industry}/{self.Name}/cashflow/quarterly.xlsx"
            my_col = list(self.income_rial_quarterly.index)
            my_col.insert(0, "Data")
        cash_flow_original = pd.read_excel(adress)
        # year = re.findall("[0123456789]{4}/[0123456789]{2}", str(cash_flow_original))
        cash = cash_flow_original[
            (cash_flow_original["Unnamed: 1"] == "نقد حاصل از عملیات")
            | (cash_flow_original["Unnamed: 1"] == "مالیات بر درامد پرداختنی")
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه دریافتنی بابت فروش دارایی های ثابت مشهود"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه دریافتی بابت فروش دارایی های نامشهود"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه پرداختی بابت تحصیل دارایی های نامشهود"
            )
            | (cash_flow_original["Unnamed: 1"] == "وجوه دریافتنی حاصل از استقراض")
            | (cash_flow_original["Unnamed: 1"] == "بازپرداخت استقراض")
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه پرداختنی بابت تحصیل دارایی های ثابت مشهود"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه دریافتنی بابت فروش سرمایه گذاری های بلند مدت"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه دریافتنی بابت فروش سرمایه گذاری های کوتاه مدت"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه پرداختنی بابت تحصیل سرمایه گذاری های کوتاه مدت"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه دریافتنی حاصل از افزایش سرمایه"
            )
            | (
                cash_flow_original["Unnamed: 1"]
                == "وجوه پرداختنی بابت تحصیل سرمایه گذاری های بلند مدت"
            )
            | (cash_flow_original["Unnamed: 1"] == "سود پرداختنی بابت استقراض")
        ]
        # my_col = year.insert(0, "Data")
        # my_col = ["Data", 1396, 1397, 1398, 139, 1400]
        cash_flow = cash.copy()
        cash_flow.dropna(axis=1, inplace=True)
        # rename columns
        for i in range(len(cash_flow.columns)):
            cash_flow.rename(columns={cash_flow.columns[i]: my_col[i]}, inplace=True)
        cash_flow.set_index("Data", inplace=True)
        my_index = [
            "cash from Operating",
            "tax",
            "receivables sale tangible",
            "paid buy tangible",
            "receivables sale intangible",
            "paid buy intangible",
            "receivables sale longterm invest",
            "paid buy longterm invest",
            "receivables sale shortterm invest",
            "paid buy shortterm invest",
            "receivables capital increase",
            "receivables borrowing",
            "paid borrowing",
            "interest borrowing",
        ]
        # rename index

        for i in range(len(my_index)):
            cash_flow.rename(index={cash_flow.index[i]: my_index[i]}, inplace=True)

        cash_flow = cash_flow.T
        # remove '-'
        for i in cash_flow.index:
            for j in cash_flow.columns:
                if cash_flow.loc[i][j] == "-":
                    cash_flow.loc[i][j] = 0.1
        # convert to nuber
        for c in cash_flow.columns:
            cash_flow[c] = [int(i) for i in cash_flow[c]]
        # create_cash_flow_com
        cash_com = pd.DataFrame(index=cash_flow.index, columns=cash_flow.columns)
        if preiode == "yearly":
            for i in cash_flow.columns:
                cash_com[i] = cash_flow[i] / self.income_rial_yearly["Net_Profit"]
        elif preiode == "quarterly":
            for i in cash_flow.columns:
                cash_com[i] = cash_flow[i] / self.income_rial_quarterly["Net_Profit"]
        # send cashflow to self
        if preiode == "yearly":
            self.cash_flow_yearly = cash_flow
            self.cash_com_yearly = cash_com
        elif preiode == "quarterly":
            self.cash_flow_quarterly = cash_flow
            self.cash_com_quarterly = cash_com
        return cash_flow

    def plot_balance_sheet(self):
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 3, 1)
        plt.plot(self.balance_com["short term receivables"], marker="o")
        plt.title("Receivables")
        plt.subplot(1, 3, 2)
        plt.plot(self.balance_com["inventory"], marker="o")
        plt.title("inventory")
        plt.subplot(1, 3, 3)
        plt.plot(self.balance_com["prepayment"], marker="o")
        plt.title("prepayment")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.balance_com["payable"], marker="o")
        plt.title("Payable")
        plt.subplot(1, 2, 2)
        plt.plot(self.balance_com["pre received"], marker="o")
        plt.title("Prereceived")
        plt.figure(figsize=[20, 8])
        plt.plot(self.balance_com["wc"], marker="o")
        plt.title("Working capital")

    def predict_balance_sheet(self):
        # call data from self
        balance = self.balance_sheet_yearly.copy()
        cost = np.abs(self.income_rial_yearly["Cost_of_Revenue"])
        tangible = self.tangible.copy()
        future_year = balance.index[-1] + 1
        receivable_rev_ratio = (
            self.balance_sheet_yearly["short term receivables"]
            / self.income_rial_yearly["Total_Revenue"]
        )

        # weight_average
        if self.industry == "siman":
            w = [1, 2]
        elif self.industry == "dode":
            w = [1, 6]
        elif self.industry == "urea":
            w = [1, 2]
        else:
            w = [1, 2]
        receivable_rev_ratio_mean = np.average(receivable_rev_ratio[-2:], weights=w)
        # add future year
        balance.loc[future_year] = np.zeros(len(balance.iloc[0]))
        balance.loc[future_year + 1] = np.zeros(len(balance.iloc[0]))
        # add short_term_receivabe to data frame
        balance.loc[future_year]["short term receivables"] = (
            receivable_rev_ratio_mean
            * self.pred_income.loc[future_year]["Total_Revenue"]
        )
        balance.loc[future_year + 1]["short term receivables"] = (
            receivable_rev_ratio_mean
            * self.pred_income.loc[future_year + 1]["Total_Revenue"]
        )
        # predict payable
        payable_cost_ratio = self.balance_sheet_yearly["payable"] / cost
        payable_cost_ratio_mean = np.average(payable_cost_ratio[-2:], weights=w)
        # add payable predict to data frame
        balance.loc[future_year]["payable"] = payable_cost_ratio_mean * np.abs(
            self.pred_income.loc[future_year]["Cost_of_Revenue"]
        )
        balance.loc[future_year + 1]["payable"] = payable_cost_ratio_mean * np.abs(
            self.pred_income.loc[future_year + 1]["Cost_of_Revenue"]
        )
        # predict inventory
        inventory_cost_ratio = self.balance_sheet_yearly["inventory"] / cost
        inventory_cost_ratio_mean = np.average(inventory_cost_ratio[-2:], weights=w)
        # add inventory to datafeame
        balance.loc[future_year]["inventory"] = (
            np.abs(self.pred_income.loc[future_year]["Cost_of_Revenue"])
            * inventory_cost_ratio_mean
        )
        balance.loc[future_year + 1]["inventory"] = (
            np.abs(self.pred_income.loc[future_year + 1]["Cost_of_Revenue"])
            * inventory_cost_ratio_mean
        )
        # predict prepayment
        prepayment_cost_ratio = self.balance_sheet_yearly["prepayment"] / cost
        prepayment_cost_ratio_mean = np.average(prepayment_cost_ratio[-2:], weights=w)
        # add prepayment to data frame
        balance.loc[future_year]["prepayment"] = prepayment_cost_ratio_mean * np.abs(
            self.pred_income.loc[future_year]["Cost_of_Revenue"]
        )
        balance.loc[future_year + 1][
            "prepayment"
        ] = prepayment_cost_ratio_mean * np.abs(
            self.pred_income.loc[future_year + 1]["Cost_of_Revenue"]
        )
        # predict prerecieved
        prerecieved_rev_ratio = (
            self.balance_sheet_yearly["pre received"]
            / self.income_rial_yearly["Total_Revenue"]
        )
        prerecieved_rev_ratio_mean = np.average(prerecieved_rev_ratio[-2:], weights=w)
        # add prerecieved to data frame
        balance.loc[future_year]["pre received"] = (
            prerecieved_rev_ratio_mean
            * self.pred_income.loc[future_year]["Total_Revenue"]
        )
        balance.loc[future_year + 1]["pre received"] = (
            prerecieved_rev_ratio_mean
            * self.pred_income.loc[future_year + 1]["Total_Revenue"]
        )
        # add future year to tangible
        tangible.loc[future_year] = np.zeros(len(tangible.iloc[0]))
        tangible.loc[future_year, "first"] = tangible.loc[future_year - 1]["end"]
        tangible.loc[future_year, "add"] = np.average(
            tangible["add_cost_ratio"][-3:-1], weights=w
        ) * np.abs(self.pred_income.loc[future_year]["Cost_of_Revenue"])
        tangible.loc[future_year, "depreciation"] = np.average(
            tangible["depreciation_ratio"][-3:-1], weights=w
        ) * (
            tangible.loc[future_year]["first"]
            + tangible.loc[future_year]["add"]
            - tangible.loc[future_year - 1]["revaluation"]
        )
        tangible.loc[future_year, "end"] = (
            tangible.loc[future_year]["first"]
            + tangible.loc[future_year]["add"]
            - tangible.loc[future_year]["depreciation"]
        )
        tangible.loc[future_year, "add_cost_ratio"] = tangible.loc[future_year][
            "add"
        ] / np.abs(self.pred_income.loc[future_year]["Cost_of_Revenue"])
        tangible.loc[future_year, "depreciation_ratio"] = tangible.loc[future_year][
            "depreciation"
        ] / (tangible.loc[future_year]["add"] + tangible.loc[future_year]["first"])
        self.pred_tangible = tangible
        # add future year + 1 to data frame
        tangible.loc[future_year + 1] = np.zeros(len(tangible.iloc[0]))
        tangible.loc[future_year + 1, "first"] = tangible.loc[future_year]["end"]
        tangible.loc[future_year + 1, "add"] = np.average(
            tangible["add_cost_ratio"][-3:-1], weights=w
        ) * np.abs(self.pred_income.loc[future_year + 1]["Cost_of_Revenue"])
        tangible.loc[future_year + 1, "depreciation"] = np.average(
            tangible["depreciation_ratio"][-3:-1], weights=w
        ) * (
            tangible.loc[future_year + 1]["first"]
            + tangible.loc[future_year + 1]["add"]
            - tangible.loc[future_year + 1 - 1]["revaluation"]
        )
        tangible.loc[future_year + 1, "end"] = (
            tangible.loc[future_year + 1]["first"]
            + tangible.loc[future_year + 1]["add"]
            - tangible.loc[future_year + 1]["depreciation"]
        )
        tangible.loc[future_year + 1, "add_cost_ratio"] = tangible.loc[future_year + 1][
            "add"
        ] / np.abs(self.pred_income.loc[future_year + 1]["Cost_of_Revenue"])
        tangible.loc[future_year + 1, "depreciation_ratio"] = tangible.loc[
            future_year + 1
        ]["depreciation"] / (
            tangible.loc[future_year + 1]["add"]
            + tangible.loc[future_year + 1]["first"]
        )

        # add tangible to data frame
        balance.loc[future_year, "tangible assets"] = tangible.loc[future_year]["end"]
        balance.loc[future_year + 1, "tangible assets"] = tangible.loc[future_year + 1][
            "end"
        ]
        # add wc to dataframe
        balance.loc[future_year]["wc"] = (
            balance.loc[future_year]["short term receivables"]
            + balance.loc[future_year]["inventory"]
            + balance.loc[future_year]["prepayment"]
        ) - (
            balance.loc[future_year]["payable"]
            + balance.loc[future_year]["pre received"]
        )
        # add wc to dataframe
        balance.loc[future_year + 1]["wc"] = (
            balance.loc[future_year + 1]["short term receivables"]
            + balance.loc[future_year + 1]["inventory"]
            + balance.loc[future_year + 1]["prepayment"]
        ) - (
            balance.loc[future_year + 1]["payable"]
            + balance.loc[future_year + 1]["pre received"]
        )
        # create_balance_pred_com
        balance_com = pd.DataFrame(columns=balance.columns, index=balance.index)
        for i in balance.columns:
            balance_com[i] = balance[i] / self.pred_income["Total_Revenue"]
        # create_inv_balance_pred
        inv_balance = pd.DataFrame(columns=balance.columns, index=balance.index)
        for i in inv_balance.columns:
            inv_balance[i] = balance[i].diff(1)
        inv_balance.dropna(axis=0, inplace=True)
        # send data to self
        self.pred_balance = balance
        self.pred_balane_com = balance_com
        self.pred_inv_balance = inv_balance
        self.tangible = tangible

    def get_cost(self, period):
        # read cost data
        if period == "yearly":
            # all data is cost_dl 
            cost_dl = pd.read_excel(
                f"{DB}/industries/{self.industry}/{self.Name}/cost/yearly.xlsx"
            )
            official_dl = pd.read_excel(
                f"{DB}/industries/{self.industry}/{self.Name}/official/yearly.xlsx"
            )
            #select desired data
            cost=select_df(cost_dl,'بهای تمام شده','جمع بهای تمام شده')
            overhead=select_df(cost_dl,'هزینه سربار','جمع')
            official=select_df(official_dl,'هزینه های عمومی و اداری','جمع')
            personnel=select_df(official_dl,'تعداد پرسنل','تعداد پرسنل تولیدی شرکت')    
            count_consump=select_df(cost_dl,'مقدار مصرف طی دوره','جمع')                  
            #define column
            my_col = list(self.income_rial_yearly.index)
            my_col.insert(0, "Data")

        elif period == "quarterly":
            cost_dl = pd.read_excel(
                f"{DB}/industries/{self.industry}/{self.Name}/cost/quarterly.xlsx"
            )
            official_dl = pd.read_excel(
                f"{DB}/industries/{self.industry}/{self.Name}/official/quarterly.xlsx"
            )            
            #select desired data
            cost=select_df(cost_dl,'بهای تمام شده','جمع بهای تمام شده')
            overhead=select_df(cost_dl,'هزینه سربار','جمع')
            official=select_df(official_dl,'هزینه های عمومی و اداری','جمع')
            personnel=select_df(official_dl,'تعداد پرسنل','تعداد پرسنل تولیدی شرکت')
            count_consump=select_df(cost_dl,'مقدار مصرف طی دوره','جمع')
            my_col = list(self.income_rial_quarterly.index)
            my_col.insert(0, "Data")
        #preprocess data
        cost.dropna(how="all", inplace=True)
        official.dropna(how="all", inplace=True)
        overhead.dropna(how="all", inplace=True)
        personnel.dropna(how="all", inplace=True)
        count_consump.dropna(how="all", inplace=True)
        # change column name
        for i in range(len(my_col)):
            cost.rename(columns={cost.columns[i]: my_col[i]}, inplace=True)
            official.rename(columns={official.columns[i]: my_col[i]}, inplace=True)
            overhead.rename(columns={overhead.columns[i]: my_col[i]}, inplace=True)
            personnel.rename(columns={personnel.columns[i]: my_col[i]}, inplace=True)
            count_consump.rename(columns={count_consump.columns[i]:my_col[i]},inplace=True)
        cost.dropna(axis=0, inplace=True)
        official.dropna(axis=0, inplace=True)
        overhead.dropna(axis=0, inplace=True)
        personnel.dropna(axis=0, inplace=True)
        count_consump.dropna(axis=0, inplace=True)
        # set Data is index 
        cost.set_index("Data", inplace=True)
        official.set_index("Data", inplace=True)
        overhead.set_index("Data", inplace=True)
        personnel.set_index("Data", inplace=True)
        count_consump.set_index('Data',inplace=True)
        # drop unnessecary data
        cost.drop('بهای تمام شده',inplace=True)
        overhead.drop('هزینه سربار',inplace=True)
        official.drop('هزینه های عمومی و اداری',inplace=True)
        personnel.drop('تعداد پرسنل',inplace=True)
        
        cost_index = [
            "direct_material",
            "direct_salary",
            "overhead",
            "total",
            "unabsorbed_cost",
            "total_cost_prod",
            "inventory_of_goods_under_costruction",
            "wastage",
            "cost_prod",
            "inventory_of_goods_first_periode",
            "inventory_of_goods_last_periode",
            "cost_of_sell_goods",
            "cost_of_service",
            "total_cost",
        ]
        # change index name to english
        for i in range(len(cost_index)):
            cost.rename(index={cost.index[i]: cost_index[i]}, inplace=True)
        # change official and overhead index to english
        official_index = [
            "transport",
            "after_sale_service",
            "commision_sell",
            "advertisingt",
            "consuming_material",
            "energy",
            "depreciation",
            "salary",
            "doubtful_claims",
            "other",
            "total",
        ]
        for i in range(len(official_index)):
            official.rename(index={official.index[i]: official_index[i]}, inplace=True)
            overhead.rename(index={overhead.index[i]: official_index[i]}, inplace=True)
        #change personel index to english
        personnel_index=['prod','non_prod']
        for i in range(len(personnel_index)):
            personnel.rename(index={personnel.index[i]:personnel_index[i]},inplace=True)
        # transpose data
        cost = cost.T
        overhead = overhead.T
        official = official.T
        personnel=personnel.T
        #add total to personel
        personnel['total']=personnel['prod']+personnel['non_prod']
        # define new definition of cost extract units of cost
        my_cost = pd.DataFrame(columns=["salary", "material", "energy"])
        #inventory ratio
        alpha=cost['total_cost']/cost['total']
        self.inventory_ratio=alpha
        my_cost["salary"] = alpha*(
            cost["direct_salary"] + official["salary"] + overhead["salary"]
        )
        my_cost["material"] =alpha* (
            cost["direct_material"]
            + overhead["consuming_material"]
            + official["consuming_material"]
        )
        my_cost["energy"] = official["energy"] + overhead["energy"]
        my_cost["depreciation"] = official["depreciation"] + overhead["depreciation"]
        my_cost["transport"] = official["transport"] + overhead["transport"]
        my_cost["other"] = overhead["other"] + official["other"]
        my_cost["total"] = (
            my_cost["salary"]
            + my_cost["material"]
            + my_cost["energy"]
            + my_cost["depreciation"]
            + my_cost["transport"]
            + my_cost["other"]
        )

        # send data to self
        if period == "yearly":
            self.cost_yearly = cost
            self.overhead_yearly = overhead
            self.official_yearly = official
            self.my_cost_yearly = my_cost
            self.personnel_yearly=personnel
            # create cost com to revenue
            my_cost_com = pd.DataFrame(columns=my_cost.columns)
            for i in my_cost:
                my_cost_com[i] = my_cost[i] / self.income_rial_yearly["Total_Revenue"]
            my_cost_com["margin"] = (
                np.ones(len(my_cost_com["total"])) - my_cost_com["total"]
            )
            self.my_cost_com_yearly = my_cost_com
            my_cost_unit = pd.DataFrame(columns=my_cost.columns)
            for i in my_cost.columns:
                my_cost_unit[i] = my_cost[i] / self.product_yearly["Count"].values
            my_cost_unit["profit"] = (
                self.product_yearly["Rate"].values - my_cost_unit["total"]
            )
            self.my_cost_unit_yearly = my_cost_unit

        elif period == "quarterly":
            self.cost_quarterly = cost
            self.overhead_quarterly = overhead
            self.official_quarterly = official
            self.my_cost_quarterly = my_cost
            self.personnel_quarterly=personnel
            # create cost com to revenue
            my_cost_com = pd.DataFrame(columns=my_cost.columns)
            for i in my_cost:
                my_cost_com[i] = my_cost[i] / self.income_rial_quarterly["Total_Revenue"]
            my_cost_com["margin"] = (
                np.ones(len(my_cost_com["total"])) - my_cost_com["total"]
            )
            my_cost_unit = pd.DataFrame(columns=my_cost.columns)
            for i in my_cost.columns:
                my_cost_unit[i] = my_cost[i] / self.product_quarterly["Count"].values
            my_cost_unit["profit"] = (
                self.product_quarterly["Rate"].values - my_cost_unit["total"]
            )
            self.my_cost_unit_quarterly = my_cost_unit
            self.my_cost_com_quarterly = my_cost_com

    def sensitivity(self):
        rate = []
        net_profit = []
        pe_fw = []
        rev = []
        p_fcfe = []
        for i in np.linspace(0.5, 2, 40):
            self.predict_income(i, 1)
            self.predict_interst()
            self.create_fcfe()
            rate.append(self.hypothesis["Last_Rate"])
            net_profit.append(self.hypothesis["Net_Profit"])
            pe_fw.append(self.pe_fw)
            p_fcfe.append(self.p_fcfe)
            rev.append(self.hypothesis["total_rev"])
        df = pd.DataFrame(columns=["rate", "net_profit"])
        df["rate"] = rate
        df["net_profit"] = net_profit
        df["pe_fw"] = pe_fw
        df["p_fcfe"] = p_fcfe
        df["rev"] = rev
        self.sensitivity = df

    def plot_cost(self):
        plt.figure(figsize=[18,15])
        self.my_cost_yearly[
            ["salary", "material", "transport", "depreciation", "energy"]
        ].T.plot(kind="pie", subplots=True, figsize=[20, 18], autopct="%.2f")

    def plot_export(self):
        try:
            self.product_yearly.iloc[1:][["Domestic", "Foreign"]].T.plot(
                kind="pie", subplots=True, figsize=[20, 8], autopct="%.2f"
            )
        except:
            print('data not available')    

    def create_interest_data(self):
        tangible = pd.DataFrame(
            columns=["first", "add", "depreciation", "end"],
            index=list(self.balance_sheet_yearly.index[1:]),
        )
        # add revaluation manually
        tangible["revaluation"] = 0
        if self.Name == "shekarbon":
            tangible.loc[1400, "revaluation"] = 6135128

        tangible["first"] = self.balance_sheet_yearly["tangible assets"][:-1].values
        tangible["end"] = self.balance_sheet_yearly["tangible assets"][1:].values
        tangible["depreciation"] = self.my_cost_yearly[1:]["depreciation"].values
        tangible["add"] = (
            tangible["end"]
            - tangible["first"]
            + tangible["depreciation"]
            - tangible["revaluation"]
        )
        # remove zero value
        for i in tangible.index:
            for j in tangible.columns:
                if tangible.loc[i, j] == 0:
                    tangible.loc[i, j] = 0.1
        tangible["add_cost_ratio"] = tangible["add"] / np.abs(
            self.income_rial_yearly[1:]["Cost_of_Revenue"].values
        )
        tangible["depreciation_ratio"] = tangible["depreciation"] / (
            tangible["add"] + tangible["first"]
        )
        self.tangible = tangible
        interest = pd.DataFrame(
            columns=["first", "add", "pay", "interest", "end"],
            index=self.income_rial_yearly.index[1:],
        )

        interest["first"] = self.balance_sheet_yearly["financial facilities"][
            :-1
        ].values
        interest["end"] = self.balance_sheet_yearly["financial facilities"][1:].values
        interest["interest"] = np.abs(
            self.income_rial_yearly["Interest_Expense"][1:].values
        )
        interest["add"] = self.cash_flow_yearly["receivables borrowing"][1:].values
        interest["pay"] = interest["first"] + interest["add"] - interest["end"]
        # remove_zero values:
        for i in interest.index:
            for j in interest.columns:
                if interest.loc[i, j] == 0:
                    interest.loc[i, j] = 0.1
        interest["add_inv_ratio"] = interest["add"] / (
            self.tangible["add"] + self.inv_balance_yearly["wc"]
        )
        #remove negative add inv ratio
        for i in interest.index:
            if interest.loc[i,'add_inv_ratio']<0:
                interest.loc[i,'add_inv_ratio']=0
        interest["pay_ratio"] = interest["pay"] / (interest["first"] + interest["add"])
        interest["interest_ratio"] = interest["interest"] / (
            interest["first"] + interest["add"]
        )
        self.interest = interest
        self.tangible = tangible

    def predict_interst(self):
        w = [1, 3]
        future_year = self.income_rial_yearly.index[-1] + 1
        self.future_year = future_year
        interest = self.interest
        # add future year to interest
        interest.loc[future_year] = np.zeros(len(interest.iloc[0]))
        interest.loc[future_year, "first"] = interest.loc[future_year - 1]["end"]
        #investing=capital expenditure+working capital
        inv = (
            self.pred_inv_balance.loc[future_year]["wc"]
            + self.tangible.loc[future_year]["add"]
        )
        if inv<0:
            inv=0
        interest.loc[future_year, "add"] = (
            np.average(interest["add_inv_ratio"][-3:-1]) * inv
        )
        interest.loc[future_year, "pay"] = np.average(interest["pay_ratio"][-3:-1]) * (
            interest.loc[future_year, "first"] + interest.loc[future_year, "add"]
        )
        interest.loc[future_year, "interest"] = np.average(
            interest["interest_ratio"][-3:-1]
        ) * (interest.loc[future_year, "first"] + interest.loc[future_year, "add"])
        interest.loc[future_year, "end"] = (
            interest.loc[future_year, "first"]
            + interest.loc[future_year, "add"]
            - interest.loc[future_year, "pay"]
        )
        # add future year+1 to interest 
        interest.loc[future_year + 1] = np.zeros(len(interest.iloc[0]))
        interest.loc[future_year + 1, "first"] = interest.loc[future_year]["end"]
        inv = (
            self.pred_inv_balance.loc[future_year + 1]["wc"]
            + self.tangible.loc[future_year + 1]["add"]
        )
        interest.loc[future_year + 1, "add"] = (
            np.average(interest["add_inv_ratio"][-3:-1]) * inv
        )
        interest.loc[future_year + 1, "pay"] = np.average(
            interest["pay_ratio"][-3:-1]
        ) * (
            interest.loc[future_year + 1, "first"]
            + interest.loc[future_year + 1, "add"]
        )
        interest.loc[future_year + 1, "interest"] = np.average(
            interest["interest_ratio"][-3:-1]
        ) * (
            interest.loc[future_year + 1, "first"]
            + interest.loc[future_year + 1, "add"]
        )
        interest.loc[future_year + 1, "end"] = (
            interest.loc[future_year + 1, "first"]
            + interest.loc[future_year + 1, "add"]
            - interest.loc[future_year + 1, "pay"]
        )
        # add interest to pred_income
        self.pred_income.loc[future_year, "Interest_Expense"] = -interest.loc[
            future_year, "interest"
        ]
        self.pred_income.loc[future_year + 1, "Interest_Expense"] = -interest.loc[
            future_year, "interest"
        ]
        # add financial facilities to pred_balance
        self.pred_balance.loc[future_year]["financial facilities"] = interest.loc[
            future_year
        ]["end"]
        self.pred_balance.loc[future_year + 1]["financial facilities"] = interest.loc[
            future_year + 1
        ]["end"]
        # add pre tax to data frame
        self.pred_income.loc[future_year, "Pretax_Income"] = (
            self.pred_income.loc[future_year]["Operating_Income"]
            + self.pred_income.loc[future_year]["Interest_Expense"]
            + self.pred_income.loc[future_year]["Other_non_operate_Income_Expense"]
        )
        self.pred_income.loc[future_year + 1, "Pretax_Income"] = (
            self.pred_income.loc[future_year + 1]["Operating_Income"]
            + self.pred_income.loc[future_year + 1]["Interest_Expense"]
            + self.pred_income.loc[future_year + 1]["Other_non_operate_Income_Expense"]
        )
        # add tax provision
        tax_ratio = (
            self.income_rial_yearly["Tax_Provision"]
            / self.income_rial_yearly["Pretax_Income"]
        )
        tax_ratio_mean = np.average(tax_ratio[-3:-1], weights=w)
        tax_pred = tax_ratio_mean * self.pred_income.loc[future_year]["Pretax_Income"]
        self.pred_income.loc[future_year, "Tax_Provision"] = tax_pred
        self.pred_income.loc[future_year + 1, "Tax_Provision"] = (
            tax_ratio_mean * self.pred_income.loc[future_year]["Pretax_Income"]
        )
        # add net_income_common
        self.pred_income.loc[future_year, "Net_Income_Common"] = (
            self.pred_income.loc[future_year]["Pretax_Income"]
            + self.pred_income.loc[future_year]["Tax_Provision"]
        )
        self.pred_income.loc[future_year + 1, "Net_Income_Common"] = (
            self.pred_income.loc[future_year + 1]["Pretax_Income"]
            + self.pred_income.loc[future_year + 1]["Tax_Provision"]
        )
        self.pred_income.loc[future_year, "Net_Profit"] = (
            self.pred_income.loc[future_year]["Pretax_Income"]
            + self.pred_income.loc[future_year]["Tax_Provision"]
        )
        self.pred_income.loc[future_year + 1, "Net_Profit"] = (
            self.pred_income.loc[future_year + 1]["Pretax_Income"]
            + self.pred_income.loc[future_year + 1]["Tax_Provision"]
        )
        # add capital to data frame
        self.pred_income.loc[future_year, "Capital"] = self.income_rial_quarterly[
            "Capital"
        ].iloc[-1]
        self.pred_income.loc[future_year + 1, "Capital"] = self.income_rial_quarterly[
            "Capital"
        ].iloc[-1]
        # add eps to data frame
        self.pred_income.loc[future_year, "EPS"] = (
            self.pred_income.loc[future_year]["Net_Profit"]
            * 1000
            / self.pred_income.loc[future_year]["Capital"]
        )
        self.pred_income.loc[future_year + 1, "EPS"] = (
            self.pred_income.loc[future_year + 1]["Net_Profit"]
            * 1000
            / self.pred_income.loc[future_year + 1]["Capital"]
        )
        self.pred_income.loc[future_year, "EPS_Capital"] = self.pred_income.loc[
            future_year
        ]["EPS"]
        self.pred_income.loc[future_year + 1, "EPS_Capital"] = self.pred_income.loc[
            future_year + 1
        ]["EPS"]

        # create pred_income_cagr
        self.pred_income_cagr = pd.DataFrame(columns=self.pred_income.columns)
        for i in self.pred_income.columns:
            self.pred_income_cagr[i] = self.pred_income[i].pct_change()
        self.pred_income_cagr.dropna(axis=0, how="all", inplace=True)
        pred_com = pd.DataFrame(
            index=self.pred_income.index, columns=self.pred_income.columns
        )
        for i in self.pred_income.index:
            pred_com.loc[i] = (
                self.pred_income.loc[i] / self.pred_income.loc[i]["Total_Revenue"]
            )
        self.pred_com = pred_com
        try:
            self.create_eps_data()
        except:
            print(f"add eps data {self.Name}")

    def create_fcfe(self):
        fcfe = pd.DataFrame(
            columns=["net_profit", "fcfe", "ratio"], index=self.pred_inv_balance.index
        )
        fcfe["net_profit"] = self.pred_income["Net_Profit"][1:].values
        fcfe["fcfe"] = (
            fcfe["net_profit"]
            + self.tangible["depreciation"]
            - self.tangible["add"]
            - self.pred_inv_balance["wc"]
        )
        fcfe["ratio"] = fcfe["fcfe"] / fcfe["net_profit"]
        self.fcfe = fcfe
        self.p_fcfe = self.pe_fw / self.fcfe.loc[self.future_year]["ratio"]

    def get_product(self, period):
        if period == "yearly":         
            #all data
            product_dl=pd.read_excel(f"{DB}/industries/{self.industry}/{self.Name}/product/yearly.xlsx")
            #select desired data
            count_product=select_df(product_dl,'مقدار تولید','جمع')
            count_revenue=select_df(product_dl,'مقدار فروش','جمع')
            price_revenue=select_df(product_dl,'مبلغ فروش','جمع')
            #define column
            my_col = list(self.income_rial_yearly.index)
            my_col.insert(0, "Data")     
            my_col.insert(1,'unit')           
          
        elif period=='monthly':
            #all data
            product_dl=pd.read_excel(f"{DB}/industries/{self.industry}/{self.Name}/product/monthly.xlsx")
            #selece desired data
            count_product=select_df(product_dl,'مقدار تولید','جمع')
            count_revenue=select_df(product_dl,'مقدار فروش','جمع')
            price_revenue=select_df(product_dl,'مبلغ فروش','جمع')
            #define my_col
            all_time_id=re.findall(regex_en_timeid_q, str(count_product.iloc[0]))
            my_col=all_time_id
            my_col.insert(0, "Data")
            my_col.insert(1,'unit') 
        #change column name
        for i in range(len(my_col)):
           count_product.rename(columns={count_product.columns[i]: my_col[i]}, inplace=True)  
           count_revenue.rename(columns={count_revenue.columns[i]: my_col[i]}, inplace=True)  
           price_revenue.rename(columns={price_revenue.columns[i]: my_col[i]}, inplace=True)  
        # set Data is index 
        count_product.set_index("Data", inplace=True) 
        count_revenue.set_index("Data", inplace=True)
        price_revenue.set_index("Data", inplace=True)
        #delete unnecessary data'
        count_product.dropna(how='all',inplace=True)
        count_product.drop('مقدار تولید',inplace=True)
        count_revenue.dropna(how='all',inplace=True)
        count_revenue.drop('مقدار فروش',inplace=True)  
        price_revenue.dropna(how='all',inplace=True) 
        price_revenue.drop('مبلغ فروش',inplace=True)     
        #transpose data      
        count_product=count_product.T
        count_revenue=count_revenue.T
        price_revenue=price_revenue.T
        #extract unit and delete from df
        self.unit_prod=count_product.loc[['unit']]
        count_product.drop('unit',inplace=True)
        count_revenue.drop('unit',inplace=True)
        price_revenue.drop('unit',inplace=True)
        #delete duplicated culoumns
        count_product=count_product.loc[:,~count_product.columns.duplicated()]
        count_revenue=count_revenue.loc[:,~count_revenue.columns.duplicated()]
        price_revenue=price_revenue.loc[:,~price_revenue.columns.duplicated()]
        #delete empty file
        delete_empty(count_product)
        delete_empty(count_revenue)
        delete_empty(price_revenue)  
        #create count_product_com
        count_product_com=pd.DataFrame(columns=count_product.columns)
        for i in count_product.columns:
            count_product_com[i]=count_product[i]/count_product['جمع']
        #create count_revenue_com    
        count_revenue_com=pd.DataFrame(columns=count_revenue.columns)
        for i in count_revenue.columns:
            count_revenue_com[i]=count_revenue[i]/count_revenue['جمع']  
        #create price_revenue_com    
        price_revenue_com=pd.DataFrame(columns=price_revenue.columns)
        for i in price_revenue.columns:
            price_revenue_com[i]=price_revenue[i]/price_revenue['جمع']                   
        ########## create product_dataframe ################
        product=pd.DataFrame(columns=['Product'])
        product['Product']=count_product['جمع']
        product['Count']=count_revenue['جمع']
        product['Revenue']=price_revenue['جمع']
        product['Rate']=product['Revenue']/product['Count']
        cycle, trend = hp_filter.hpfilter(product["Count"])
        product["cycle"] = cycle
        product["trend"] = trend        
        ####### create quarterly data from monthly #########
        if period == "monthly":

            # create quarterly product
            fiscal_dic = {
                12: {1: 3, 2: 6, 3: 9, 4: 12},
                9: {1: 12, 2: 3, 3: 6, 4: 9},
                6: {1: 9, 2: 12, 3: 3, 4: 6},
                3: {1: 6, 2: 9, 3: 12, 4: 3},
                10: {1: 1, 2: 4, 3: 7, 4: 10},
            }

            q_index = []
            for i in range(len(product.index)):
                if int(product.index[i].split("/")[1]) % 3 == 0:
                    q_index.append(i)
            product_q = pd.DataFrame(columns=["Product", "Count", "Revenue"])
            prod = []
            count = []
            rev = []
            for i in q_index:
                prod.append(product.iloc[i - 2 : i + 1]["Product"].sum())
                count.append(product.iloc[i - 2 : i + 1]["Count"].sum())
                rev.append(product.iloc[i - 2 : i + 1]["Revenue"].sum())
            product_q["Product"] = prod
            product_q["Count"] = count
            product_q["Revenue"] = rev
            product_q["Rate"] = product_q["Revenue"] / product_q["Count"]
            # lag from statement and product report
            last_m = int(product.index[-1].split("/")[1])
            last_q = int(self.income_rial_quarterly.index[-1][-1])
            if (last_m % 3 == 0) & (last_m > fiscal_dic[self.fiscal_year][last_q]):
                product_q = product_q[-(self.n + 1) : -1]
            else:
                product_q = product_q[-self.n :]
            product_q.set_index(self.income_rial_quarterly.index, inplace=True)
            self.product_quarterly = product_q
        ############ send data to self ############
        if period=='yearly':
            self.count_product_yearly=count_product
            self.count_product_com_yearly=count_product_com
            
            self.count_revenue_yearly=count_revenue
            self.count_revenue_com_yearly=count_revenue_com

            self.price_revenue_yearly=price_revenue
            self.price_revenue_com_yearly=price_revenue_com 

            self.product_yearly=product           
        
        elif period=='monthly':
            self.count_product_monthly=count_product
            self.count_product_com_monthly=count_product_com
            
            self.count_revenue_monthly=count_revenue
            self.count_revenue_com_monthly=count_revenue_com

            self.price_revenue_monthly=price_revenue
            self.price_revenue_com_monthly=price_revenue_com

            self.product_monthly=product            
        return product

    def update_predict(
        self,
        alpha_prod_update=1,
        alpha_rate_update=1,
        alpha_prod_next_update=1,
        alpha_rate_next_update=1,
        salary_g_update=1,
        material_g_update=1,
        energy_g_update=1,
        dep_g_update=1,
        transport_g_update=1,
        other_g_update=1,
        salary_g_next_update=1,
        material_g_next_update=1,
        energy_g_next_update=1,
        dep_g_next_update=1,
        transport_g_next_update=1,
        other_g_next_update=1,
    ):
        self.predict_income(
            alpha_rate_update,
            alpha_prod_update,
            alpha_prod_next_update,
            alpha_rate_next_update,
            salary_g_update,
            material_g_update,
            energy_g_update,
            dep_g_update,
            transport_g_update,
            other_g_update,
            salary_g_next_update,
            material_g_next_update,
            energy_g_next_update,
            dep_g_next_update,
            transport_g_next_update,
            other_g_next_update,
        )
        self.predict_balance_sheet()
        self.predict_interst()
        self.create_fcfe()

    def plot_margin(self):
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        self.pred_cost_com["profit"].plot(marker="o")
        plt.title("margin")
        plt.subplot(1, 2, 2)
        self.product_monthly[
            -(self.fiscal_year - int(JalaliDate.today().isoformat().split("-")[1])) :
        ]["Rate"].plot(marker="o")
        plt.title("rate")

    def create_macro(self):
        # macro economic data
        macro = Macro()
        df = macro.exchange[["dollar", "cpi", "cash"]][-(self.n+1):]
        df["total_cost"] = self.my_cost_unit_yearly["total"]
        df["profit"] = self.my_cost_unit_yearly["profit"]
        df_ret = macro.exchange_ret[["dollar", "cpi", "cash"]][-(self.n+1):]
        df_ret["profit"] = self.my_cost_unit_yearly["profit"].pct_change()
        df_ret["total_cost"] = self.my_cost_unit_yearly["total"].pct_change()
        df_ret.dropna(inplace=True)
        self.macro = df
        self.macro_ret = df_ret
        # return net profit and market
        # compare returns
        mean_dollar = []
        mean_market = []
        for i in range(self.last_year - self.n + 1, self.last_year + 1):
            date_1 = pd.to_datetime(JalaliDate(i, 1, 1).to_gregorian())
            date_2 = pd.to_datetime(JalaliDate(i, 12, 29).to_gregorian())
            mean_dollar.append(self.dollar_nima[date_1:date_2]["Close"].mean())
            mean_market.append(self.Price[date_1:date_2]["Close"].mean())
        self.mean_dollar = mean_dollar
        data = pd.DataFrame(columns=["Net_Profit", "dollar"])
        data["Net_Profit"] = self.income_rial_yearly["Net_Profit"]
        data["Net_margin"] = self.income_common_rial_yearly["Net_Profit"]
        data["dollar"] = mean_dollar
        data["Market"] = mean_market
        data['cpi'] = df['cpi']
        # add future year to data
        data.loc[self.future_year] = np.zeros(len(data.iloc[0]))
        data.loc[self.future_year, "dollar"] = macro.dollar_azad.iloc[-1]["Close"]
        data.loc[self.future_year, "Net_Profit"] = self.pred_income.loc[
            self.future_year, "Net_Profit"
        ]
        data.loc[self.future_year, "Market"] = self.Price.iloc[-1]["Close"]
        # add ret and cret to data
        data["dollar_ret"] = np.log(data["dollar"] / data["dollar"].shift(1))
        data["Market_ret"] = np.log(data["Market"] / data["Market"].shift(1))
        data["dollar_cret"] = data["dollar_ret"].cumsum().apply(np.exp)
        data["Market_cret"] = data["Market_ret"].cumsum().apply(np.exp)
        data["Net_Profit_cret"] = data["Net_Profit"] / np.abs(
            data["Net_Profit"].iloc[0]
        )
        data["Net_Profit_ret"] = data["Net_Profit"].pct_change()
        data['cpi_ret']=data['cpi'].pct_change()
        data.loc[self.future_year,'cpi_ret']=0.48
        compare_ret = data[["Net_Profit_ret", "Market_ret", "dollar_ret",'cpi_ret']]
        compare_ret.dropna(inplace=True)
        self.compare_ret = compare_ret
        self.data=data

        self.dollar_income = data

    def plot_compare(self):
        plt.figure(figsize=[20,15])
        plt.subplot(2,2,1)
        plt.plot(self.compare_ret['Market_ret'],color='black',marker='o',label='Market_Ret')
        plt.bar(height=self.compare_ret['Net_Profit_ret'],x=self.compare_ret.index,alpha=0.3,label='Net_Profit_Ret')
        plt.legend()
        plt.subplot(2,2,2)
        plt.plot(self.compare_ret['Market_ret'],color='black',marker='o',label='Market_Ret') 
        plt.bar(height=self.compare_ret['cpi_ret'],x=self.compare_ret.index,alpha=0.3,label='Inflation')
        plt.legend()
        plt.subplot(2,2,3)
        plt.plot(self.compare_ret['Net_Profit_ret'],color='black',marker='o',label='Net_Profit_ret')
        plt.bar(height=self.compare_ret['cpi_ret'],x=self.compare_ret.index,alpha=0.3,label='Inflation')
        plt.legend()

    def predict_value(self,n,g=0.2,k=0.35):
        eps1=self.pred_income.loc[self.future_year,'EPS_Capital']
        eps2=self.pred_income.loc[self.future_year+1,'EPS_Capital']
        value_d=eps1/(1+k)**n+eps2/(1+k)**(1+n)
        pe_terminal=(1+g)/(k-g)
        terminal_value=(eps2*(1+g)/(k-g))/((1+k)**(1+n))
        value=value_d+terminal_value
        self.value_d=value_d
        self.terminal_value=terminal_value
        self.value=value
        self.pe_terminal=pe_terminal
    #save your analyse
    def save_analyse(self,name):
        '''
        save your analyse in analyse/name
        '''
        with open(f'{DB}/industries/{self.industry}/{self.Name}/analyse/{name}.pkl','wb') as f:
            pickle.dump(self,f)


class OptPort:
    def __init__(self, stocks, y_s=1400, m_s=1, d_s=1, y_e=1401, m_e=6, d_e=1):
        self.names = [i.Name for i in stocks]
        self.prices = [i.Price for i in stocks]
        self.start = pd.to_datetime(JalaliDate(y_s, m_s, d_s).to_gregorian())
        self.end = pd.to_datetime(JalaliDate(y_e, m_e, d_e).to_gregorian())
        self.price, self.ret, self.cret, self.mu, self.sigma = self.create_data()

    def create_data(self):
        data_price = pd.DataFrame(columns=self.names)
        data_ret = pd.DataFrame(columns=self.names)
        data_cret = pd.DataFrame(columns=self.names)
        for i in range(len(self.names)):
            data_price[self.names[i]] = self.prices[i]["Close"][self.start : self.end]
        data_price.dropna(inplace=True)
        for i in data_price.columns:
            data_ret[i] = np.log(data_price[i] / data_price[i].shift(1))
        for i in data_price.columns:
            data_cret[i] = data_ret[i].cumsum().apply(np.exp)
        data_ret.dropna(inplace=True)
        data_cret.dropna(inplace=True)
        data_price.dropna(inplace=True)
        mu = data_ret.mean().to_frame()
        sigma = data_ret.cov()
        return data_price, data_ret, data_cret, mu, sigma

    def create_random_weight(self):
        w = np.random.uniform(0, 1, (len(self.names), 1))
        w = w / sum(w)
        return w

    def measure_reward(self, w):
        expect_ret = np.dot(w.T, self.mu)[0, 0]
        risk = np.dot(np.dot(w.T, self.sigma), w)[0, 0]
        return expect_ret, risk

    def create_port(self, w):
        port = pd.DataFrame(columns=["Ret", "Cret"])
        port["Ret"] = w[0] * self.ret[self.names[0]].copy()
        for i in range(1, len(self.names)):
            port["Ret"] = port["Ret"] + w[i] * self.ret[self.names[i]]
        port["Cret"] = port["Ret"].cumsum().apply(np.exp)
        return port

    def create_fron(self):
        weight_data = []
        reward_data = []
        risk_data = []
        w1 = []
        w2 = []
        w3 = []
        w4 = []
        for i in range(10000):
            w = self.create_random_weight()
            weight_data.append(w)
            w1.append(w[0][0])
            w2.append(w[1][0])
            w3.append(w[2][0])
            w4.append(w[3][0])
            ret, risk = self.measure_reward(w)
            reward_data.append(ret)
            risk_data.append(risk)
        resault = pd.DataFrame(columns=["Ret", "Risk"])
        resault["Ret"] = reward_data
        resault["Risk"] = risk_data
        resault["w1"] = w1
        resault["w2"] = w2
        resault["w3"] = w3
        resault["w4"] = w4
        plt.figure(figsize=[20, 8])
        plt.scatter(x=resault["Risk"], y=resault["Ret"])
        plt.title("Risk_Return_model")
        plt.xlabel("Risk")
        plt.ylabel("Ret")
        for i in range(len(self.names)):
            w = np.zeros((len(self.names), 1))
            w[i, 0] = 1
            ret, risk = self.measure_reward(w)
            plt.scatter(x=risk, y=ret, s=50, marker="x", c="red")
            plt.text(risk, ret, self.names[i], fontdict={"size": 14})

        return resault

    def cost_function(self, w, M0=0.0001, penalty=10000):
        # w=w/w.sum()
        expect_ret, risk = self.measure_reward(w)
        cost = risk
        if expect_ret < M0:
            cost += penalty * (M0 - expect_ret)
        return cost

    def optimize_weight(self, M0):
        LB = np.zeros(len(self.names))
        UB = np.ones(len(self.names))
        w, best_loss = ps.pso(self.cost_function, LB, UB, swarmsize=40, maxiter=60)
        return w, best_loss


class IndustryPe:
    def __init__(self):
        self.bank_pe, self.bank_pe_n, self.bank_pe_u = get_pe_data("bank", "index")
        self.cement_pe, self.cement_pe_n, self.cement_pe_u = get_pe_data(
            "cement", "index"
        )
        self.palayesh_pe, self.palayesh_pe_n, self.palayesh_pe_u = get_pe_data(
            "palayesh", "index"
        )
        self.folad_pe, self.folad_pe_n, self.folad_pe_u = get_pe_data("folad", "index")
        self.zeraat_pe, self.zeraat_pe_n, self.zeraat_pe_u = get_pe_data(
            "zeraat", "index"
        )
        self.lastic_pe, self.lastic_pe_n, self.lastic_pe_u = get_pe_data(
            "lastic", "index"
        )
        self.daro_pe, self.daro_pe_n, self.daro_pe_u = get_pe_data("daro", "index")
        self.ghand_pe, self.ghand_pe_n, self.ghand_pe_u = get_pe_data("ghand", "index")

    def plot_pe(self):
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.bank_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(self.bank_pe["P/E-ttm"].median(), color="black", linestyle="dashed")
        plt.axvline(self.bank_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.bank_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("all_bank_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.bank_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.bank_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.bank_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.bank_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("Normall_bank_pe_data")
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.cement_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.cement_pe["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.cement_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.cement_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("all_cement_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.cement_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.cement_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.cement_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.cement_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("Normall_cement_pe_data")
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.palayesh_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.palayesh_pe["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.palayesh_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(
            self.palayesh_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed"
        )
        plt.title("all_palayeshgah_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.palayesh_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.palayesh_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.palayesh_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(
            self.palayesh_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed"
        )
        plt.title("Normall_palayeshgah_pe_data")
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.folad_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.folad_pe["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.folad_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.folad_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("all_folad_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.folad_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.folad_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.folad_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.folad_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("Normall_folad_pe_data")
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.lastic_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.lastic_pe["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.lastic_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.lastic_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("all_lastic_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.lastic_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.lastic_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.lastic_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.lastic_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("Normall_lastic_pe_data")
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.daro_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(self.daro_pe["P/E-ttm"].median(), color="black", linestyle="dashed")
        plt.axvline(self.daro_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.daro_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("all_daro_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.daro_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.daro_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.daro_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.daro_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("Normall_daro_pe_data")
        plt.figure(figsize=[15, 10])
        plt.subplot(2, 1, 1)
        plt.hist(self.ghand_pe["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.ghand_pe["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.ghand_pe["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.ghand_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("all_ghand_pe_data")
        plt.subplot(2, 1, 2)
        plt.hist(self.ghand_pe_n["P/E-ttm"], edgecolor="black", bins=100)
        plt.axvline(
            self.ghand_pe_n["P/E-ttm"].median(), color="black", linestyle="dashed"
        )
        plt.axvline(self.ghand_pe_n["P/E-ttm"].iloc[0], color="red")
        plt.axvline(self.ghand_pe.iloc[0]["P/E-ttm"], color="red", linestyle="dashed")
        plt.title("Normall_ghand_pe_data")


class Industry:
    def __init__(self, stocks):
        self.stocks = stocks
        self.net_profit_yearly = pd.DataFrame(columns=[stocks[0].Name])
        self.pe_fw = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_profit_yearly = pd.DataFrame(columns=[stocks[0].Name])
        self.revenue_yearly = pd.DataFrame(columns=[stocks[0].Name])
        self.net_margin_yearly = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_margin_yearly = pd.DataFrame(columns=[stocks[0].Name])
        self.net_profit_quarterly_12 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_profit_quarterly_12 = pd.DataFrame(columns=[stocks[0].Name])
        self.revenue_quarterly_12 = pd.DataFrame(columns=[stocks[0].Name])
        self.net_margin_quarterly_12 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_margin_quarterly_12 = pd.DataFrame(columns=[stocks[0].Name])
        self.total_industry = pd.DataFrame(
            columns=["Revenue", "Gross_Profit", "Net_Profit"]
        )
        self.total_industry_quarterly_12 = pd.DataFrame(
            columns=["Revenue", "Gross_Profit", "Net_Profit"]
        )
        temp_rev = 0
        temp_net = 0
        temp_gross = 0
        temp_rev_f = 0
        temp_net_f = 0
        temp_gross_f = 0
        self.net_margin_quarterly_9 = pd.DataFrame(columns=[stocks[0].Name])
        self.net_margin_quarterly_6 = pd.DataFrame(columns=[stocks[0].Name])
        self.net_margin_quarterly_3 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_margin_quarterly_9 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_margin_quarterly_6 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_margin_quarterly_3 = pd.DataFrame(columns=[stocks[0].Name])
        self.revenue_quarterly_9 = pd.DataFrame(columns=[stocks[0].Name])
        self.revenue_quarterly_6 = pd.DataFrame(columns=[stocks[0].Name])
        self.revenue_quarterly_3 = pd.DataFrame(columns=[stocks[0].Name])
        self.net_profit_quarterly_9 = pd.DataFrame(columns=[stocks[0].Name])
        self.net_profit_quarterly_6 = pd.DataFrame(columns=[stocks[0].Name])
        self.net_profit_quarterly_3 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_profit_quarterly_9 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_profit_quarterly_6 = pd.DataFrame(columns=[stocks[0].Name])
        self.gross_profit_quarterly_3 = pd.DataFrame(columns=[stocks[0].Name])
        self.pe = pd.DataFrame(columns=[stocks[0].Name])
        self.potential_price_g = pd.DataFrame(columns=[stocks[0].Name])
        self.pe_med = pd.DataFrame(columns=[stocks[0].Name])
        self.rate_monthly = pd.DataFrame(columns=[stocks[0].Name])
        for i in stocks:
            self.net_profit_yearly[i.Name] = i.income_rial_yearly[
                "Net_Profit"
            ] / np.abs(i.income_rial_yearly["Net_Profit"].iloc[0])
            self.gross_profit_yearly[i.Name] = i.income_rial_yearly[
                "Gross_Profit"
            ] / np.abs(i.income_rial_yearly["Gross_Profit"].iloc[0])
            self.revenue_yearly[i.Name] = i.income_rial_yearly[
                "Total_Revenue"
            ] / np.abs(i.income_rial_yearly["Total_Revenue"].iloc[0])
            self.net_margin_yearly[i.Name] = i.income_common_rial_yearly["Net_Profit"]
            self.gross_margin_yearly[i.Name] = i.income_common_rial_yearly[
                "Gross_Profit"
            ]
            self.pe_fw[i.Name] = [i.pe_fw]
            self.potential_price_g[i.Name] = [i.potential_price_g]
            self.pe_med[i.Name] = [i.pe_med]
            self.rate_monthly[i.Name] = i.product_monthly["Rate"]
            if i.fiscal_year == 12:
                self.net_profit_quarterly_12[i.Name] = i.income_rial_quarterly[
                    "Net_Profit"
                ] / np.abs(i.income_rial_quarterly["Net_Profit"].iloc[0])
                self.gross_profit_quarterly_12[i.Name] = i.income_rial_quarterly[
                    "Gross_Profit"
                ] / np.abs(i.income_rial_quarterly["Gross_Profit"].iloc[0])
                self.revenue_quarterly_12[i.Name] = i.income_rial_quarterly[
                    "Total_Revenue"
                ] / np.abs(i.income_rial_quarterly["Total_Revenue"].iloc[0])
                self.net_margin_quarterly_12[i.Name] = i.income_common_rial_quarterly[
                    "Net_Profit"
                ]
                self.gross_margin_quarterly_12[i.Name] = i.income_common_rial_quarterly[
                    "Gross_Profit"
                ]
                self.net_margin_quarterly_12.dropna(axis=1, inplace=True)
                self.gross_margin_quarterly_12.dropna(axis=1, inplace=True)
                self.revenue_quarterly_12.dropna(axis=1, inplace=True)
                self.gross_profit_quarterly_12.dropna(axis=1, inplace=True)
                self.net_profit_quarterly_12.dropna(axis=1, inplace=True)
                temp_rev_f += i.income_rial_quarterly["Total_Revenue"]
                temp_gross_f += i.income_rial_quarterly["Gross_Profit"]
                temp_net_f += i.income_rial_quarterly["Net_Profit"]
            if i.fiscal_year == 9:
                self.net_profit_quarterly_9[i.Name] = i.income_rial_quarterly[
                    "Net_Profit"
                ] / np.abs(i.income_rial_quarterly["Net_Profit"].iloc[0])
                self.gross_profit_quarterly_9[i.Name] = i.income_rial_quarterly[
                    "Gross_Profit"
                ] / np.abs(i.income_rial_quarterly["Gross_Profit"].iloc[0])
                self.revenue_quarterly_9[i.Name] = i.income_rial_quarterly[
                    "Total_Revenue"
                ] / np.abs(i.income_rial_quarterly["Total_Revenue"].iloc[0])
                self.net_margin_quarterly_9[i.Name] = i.income_common_rial_quarterly[
                    "Net_Profit"
                ]
                self.gross_margin_quarterly_9[i.Name] = i.income_common_rial_quarterly[
                    "Gross_Profit"
                ]
                self.net_margin_quarterly_9.dropna(axis=1, inplace=True)
                self.gross_margin_quarterly_9.dropna(axis=1, inplace=True)
                self.revenue_quarterly_9.dropna(axis=1, inplace=True)
                self.gross_profit_quarterly_9.dropna(axis=1, inplace=True)
                self.net_profit_quarterly_9.dropna(axis=1, inplace=True)
            if i.fiscal_year == 6:
                self.net_profit_quarterly_6[i.Name] = i.income_rial_quarterly[
                    "Net_Profit"
                ] / np.abs(i.income_rial_quarterly["Net_Profit"].iloc[0])
                self.gross_profit_quarterly_6[i.Name] = i.income_rial_quarterly[
                    "Gross_Profit"
                ] / np.abs(i.income_rial_quarterly["Gross_Profit"].iloc[0])
                self.revenue_quarterly_6[i.Name] = i.income_rial_quarterly[
                    "Total_Revenue"
                ] / np.abs(i.income_rial_quarterly["Total_Revenue"].iloc[0])
                self.net_margin_quarterly_6[i.Name] = i.income_common_rial_quarterly[
                    "Net_Profit"
                ]
                self.gross_margin_quarterly_6[i.Name] = i.income_common_rial_quarterly[
                    "Gross_Profit"
                ]
                self.net_margin_quarterly_6.dropna(axis=1, inplace=True)
                self.gross_margin_quarterly_6.dropna(axis=1, inplace=True)
                self.revenue_quarterly_6.dropna(axis=1, inplace=True)
                self.gross_profit_quarterly_6.dropna(axis=1, inplace=True)
                self.net_profit_quarterly_6.dropna(axis=1, inplace=True)
            if i.fiscal_year == 3:
                self.net_profit_quarterly_3[i.Name] = i.income_rial_quarterly[
                    "Net_Profit"
                ] / np.abs(i.income_rial_quarterly["Net_Profit"].iloc[0])
                self.gross_profit_quarterly_3[i.Name] = i.income_rial_quarterly[
                    "Gross_Profit"
                ] / np.abs(i.income_rial_quarterly["Gross_Profit"].iloc[0])
                self.revenue_quarterly_3[i.Name] = i.income_rial_quarterly[
                    "Total_Revenue"
                ] / np.abs(i.income_rial_quarterly["Total_Revenue"].iloc[0])
                self.net_margin_quarterly_3[i.Name] = i.income_common_rial_quarterly[
                    "Net_Profit"
                ]
                self.gross_margin_quarterly_3[i.Name] = i.income_common_rial_quarterly[
                    "Gross_Profit"
                ]
                self.net_margin_quarterly_3.dropna(axis=1, inplace=True)
                self.gross_margin_quarterly_3.dropna(axis=1, inplace=True)
                self.revenue_quarterly_3.dropna(axis=1, inplace=True)
                self.gross_profit_quarterly_3.dropna(axis=1, inplace=True)
                self.net_profit_quarterly_3.dropna(axis=1, inplace=True)
            temp_rev += i.income_rial_yearly["Total_Revenue"]
            temp_net += i.income_rial_yearly["Net_Profit"]
            temp_gross += i.income_rial_yearly["Net_Profit"]
            self.pe[i.Name] = i.pe["P/E-ttm"]
        self.total_industry["Revenue"] = temp_rev
        self.total_industry["Net_Profit"] = temp_net
        self.total_industry["Gross_Profit"] = temp_gross
        self.total_industry["Net_margin"] = (
            self.total_industry["Net_Profit"] / self.total_industry["Revenue"]
        )
        self.total_industry["Gross_margin"] = (
            self.total_industry["Gross_Profit"] / self.total_industry["Revenue"]
        )
        self.total_industry_quarterly_12["Revenue"] = temp_rev_f
        self.total_industry_quarterly_12["Net_Profit"] = temp_net_f
        self.total_industry_quarterly_12["Gross_Profit"] = temp_gross_f
        self.total_industry_quarterly_12["Net_margin"] = (
            self.total_industry_quarterly_12["Net_Profit"]
            / self.total_industry_quarterly_12["Revenue"]
        )
        self.total_industry_quarterly_12["Gross_margin"] = (
            self.total_industry_quarterly_12["Gross_Profit"]
            / self.total_industry_quarterly_12["Revenue"]
        )
        self.total_industry_common = self.total_industry / self.total_industry.iloc[0]

    def plot_industry(self):
        n = []
        for i in self.stocks:
            n.append(i.Name)
        pe_fw = np.squeeze(self.pe_fw.values)
        pe_med = np.squeeze(self.pe_med.values)
        price_ret = np.squeeze(self.potential_price_g)
        x_axis = np.arange(len(n))
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.bar(x_axis + 0.2, height=pe_fw, label="pe_fw", width=0.2)
        plt.bar(x_axis - 0.2, height=pe_med, label="pe_med", width=0.2)
        plt.xticks(x_axis, n)
        plt.legend()
        plt.title("pe_fw and pe_med compare")
        plt.subplot(1, 2, 2)
        plt.bar(x=n, height=price_ret, width=0.3)
        plt.title("Potential of return price")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(
            self.gross_margin_yearly, label=self.gross_margin_yearly.columns, marker="o"
        )
        plt.legend()
        plt.title("Gross_margin_yearly")
        plt.subplot(1, 2, 2)
        plt.plot(
            self.net_margin_yearly, label=self.net_margin_yearly.columns, marker="o"
        )
        plt.legend()
        plt.title("Net_margin_yearly")
        self.rate_monthly.plot(figsize=[20, 8], marker="o")
        plt.title("Monthly_Rate")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.total_industry["Net_margin"], marker="o")
        plt.title("Net_Margin")
        plt.subplot(1, 2, 2)
        plt.plot(self.total_industry["Gross_margin"], marker="o")
        plt.title("Gross_margin")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.total_industry["Revenue"], marker="o")
        plt.title("Totla_Rev")
        plt.subplot(1, 2, 2)
        plt.plot(self.total_industry["Net_Profit"], marker="o")
        plt.title("Total_net")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.total_industry_quarterly_12["Net_margin"], marker="o")
        plt.title("Totlal_Net_Margin_Fassli")
        plt.subplot(1, 2, 2)
        plt.plot(self.total_industry_quarterly_12["Gross_margin"], marker="o")
        plt.title("Gross_margin")
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 2, 1)
        plt.plot(self.total_industry_quarterly_12["Revenue"], marker="o")
        plt.title("Total_Rev_quarterly")
        plt.subplot(1, 2, 2)
        plt.plot(self.total_industry_quarterly_12["Net_Profit"], marker="o")
        plt.title("Total_Net_quarterly")
