import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

from fredapi import Fred
from statsmodels.tsa.filters import hp_filter

from statics.setting import *

plt.style.use("seaborn")
fred_apikey = "8bb5a2c9d4aac422def9c007fade8795"
fred = Fred(api_key=fred_apikey)


def get_fed_data(data):
    a = fred.get_series(data)
    df = a.to_frame()
    df.rename(columns={0: "Data"}, inplace=True)
    df["Change"] = df["Data"].pct_change() * 100
    return df


def get_forex_daily(symbol, exchange, start, end):
    api_key = "1XFLCK3MK977L2VM"
    api_url = (
        url
    ) = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={symbol}&to_symbol={exchange}&apikey={api_key}&outputsize=full"
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df["Time Series FX (Daily)"]).T
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        }
    )
    df.index = pd.to_datetime(df.index)
    for i in df.columns:
        df[i] = df[i].astype(float)
    df = df.iloc[::-1]
    df = df[start:end]
    df["Diff"] = (df["Close"] - df["Open"]) * 10000
    df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Cret"] = df["Ret"].cumsum().apply(np.exp)
    df["Cummax"] = df["Cret"].cummax()
    df["Droe_Down"] = df["Cummax"] - df["Cret"]
    return df


def get_forex_intraday(symbol, exchange, interval):
    api_key = "1XFLCK3MK977L2VM"
    api_url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={symbol}&to_symbol={exchange}&interval={interval}&apikey={api_key}&outputsize=full"
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df[f"Time Series FX ({interval})"]).T
    df.index = pd.to_datetime(df.index)
    df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        },
        inplace=True,
    )
    for i in df.columns:
        df[i] = df[i].astype(float)
    df = df.iloc[::-1]
    df["Diff"] = (df["Close"] - df["Open"]) * 10000
    df["Ret"] = df["Close"].pct_change() * 100
    return df


def corrolation(symbol1, symbol2, col1, col2, start, end):
    plt.figure(figsize=[10, 5])
    plt.scatter(
        symbol1.loc[
            symbol1.loc[start:end].index.intersection(symbol2.loc[start:end].index)
        ][col1],
        symbol2.loc[
            symbol2.loc[start:end].index.intersection(symbol1.loc[start:end].index)
        ][col2],
    )
    return np.corrcoef(
        symbol1.loc[
            symbol1.loc[start:end].index.intersection(symbol2.loc[start:end].index)
        ][col1],
        symbol2.loc[
            symbol2.loc[start:end].index.intersection(symbol1.loc[start:end].index)
        ][col2],
    )


def get_forex_Weekly(symbol, exchange):
    api_key = "1XFLCK3MK977L2VM"
    api_url = (
        url
    ) = f"https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol={symbol}&to_symbol={exchange}&apikey={api_key}&outputsize=full"
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df["Time Series FX (Weekly)"]).T
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        }
    )
    df.index = pd.to_datetime(df.index)
    for i in df.columns:
        df[i] = df[i].astype(float)
    df = df.iloc[::-1]
    df["Diff"] = (df["Close"] - df["Open"]) * 10000
    df["Ret"] = df["Close"].pct_change() * 100
    return df


def get_forex_Monthly(symbol, exchange):
    api_key = "1XFLCK3MK977L2VM"
    api_url = (
        url
    ) = f"https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol={symbol}&to_symbol={exchange}&apikey={api_key}&outputsize=full"
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df["Time Series FX (Monthly)"]).T
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        }
    )
    df.index = pd.to_datetime(df.index)
    for i in df.columns:
        df[i] = df[i].astype(float)
    df = df.iloc[::-1]
    df["Diff"] = (df["Close"] - df["Open"]) * 10000
    df["Ret"] = df["Close"].pct_change() * 100
    return df


def get_stock_Daily(symbol, start, end):
    df = yf.download(symbol)
    df = df[start:end].copy()
    df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Cret"] = df["Ret"].cumsum().apply(np.exp)
    df["Cummax"] = df["Cret"].cummax()
    return df


def get_stock_intraday(symbol, interval):
    df = yf.download(symbol, interval)
    return df


class Economy:
    def __init__(self, country, start="2010", end="2022"):
        self.country = country
        self.start = start
        self.end = end
        self.country = country
        ##### get economic_index ####
        self.read_index()

    def pmi_score(self):
        up_triger = 0
        down_triger = 0
        counter_up = 0
        counter_down = 0
        for i in range(len(self.PMI.index)):

            if self.PMI["PMI"].iloc[i] < 50:
                counter_up = 0
            else:
                counter_down = 0
            if (self.PMI["PMI"].iloc[i] > 50) & (
                self.PMI["PMI"].iloc[i] > self.PMI["PMI"].iloc[i - 1]
            ):
                self.PMI["Score"].iloc[i] = 5 + min(3, 0.3 * self.PMI["Change"].iloc[i])
            if (self.PMI["PMI"].iloc[i] > 50) & (
                self.PMI["PMI"].iloc[i] < self.PMI["PMI"].iloc[i - 1]
            ):
                counter_up += 1
                if counter_up == 1:
                    self.PMI["Score"].iloc[i] = -10
                else:
                    self.PMI["Score"].iloc[i] = 5 - min(
                        5, abs(0.5 * self.PMI["Change"].iloc[i])
                    )
            if (self.PMI["PMI"].iloc[i] < 50) & (
                self.PMI["PMI"].iloc[i] < self.PMI["PMI"].iloc[i - 1]
            ):
                self.PMI["Score"].iloc[i] = -5 - min(
                    3, abs(0.3 * self.PMI["Change"].iloc[i])
                )
            if (self.PMI["PMI"].iloc[i] < 50) & (
                self.PMI["PMI"].iloc[i] > self.PMI["PMI"].iloc[i - 1]
            ):
                counter_down += 1
                if counter_down == 1:
                    self.PMI["Score"].iloc[i] = 10
                else:
                    self.PMI["Score"].iloc[i] = -5 + min(
                        5, abs(0.5 * self.PMI["Change"].iloc[i])
                    )

    def m2_score(self):
        self.M2["Score"] = np.zeros(len(self.M2.index))
        for i in range(len(self.M2.index)):
            if (self.M2["Annualized_Change"].iloc[i] > self.M2_mean) & (
                self.M2["Annualized_Change"].iloc[i] < self.M2_m1
            ):
                a = 5 / (self.M2_m1 - self.M2_mean)
                b = -a * self.M2_mean
                self.M2["Score"].iloc[i] = a * self.M2["Annualized_Change"].iloc[i] + b
            if (self.M2["Annualized_Change"].iloc[i] > self.M2_m1) & (
                self.M2["Annualized_Change"].iloc[i] < self.M2_m2
            ):
                a = 5 / (self.M2_m2 - self.M2_m1)
                b = 10 - a * self.M2_m2
                self.M2["Score"].iloc[i] = a * self.M2["Annualized_Change"].iloc[i] + b
            if self.M2["Annualized_Change"].iloc[i] > self.M2_m2:
                a = -5 / (self.M2_m3 - self.M2_m2)
                b = 10 - a * self.M2_m2
                self.M2["Score"].iloc[i] = a * self.M2["Annualized_Change"].iloc[i] + b
            if (self.M2["Annualized_Change"].iloc[i] < self.M2_mean) & (
                self.M2["Annualized_Change"].iloc[i] > self.M2_m_2
            ):
                a = (-10) / (self.M2_m_2 - self.M2_mean)
                b = -a * self.M2_mean
                self.M2["Score"].iloc[i] = a * self.M2["Annualized_Change"].iloc[i] + b
            if self.M2["Annualized_Change"].iloc[i] < self.M2_m_2:
                a = 10 / (self.M2_m_3 - self.M2_m_2)
                b = -a * self.M2_m_2
                self.M2["Score"].iloc[i] = a * self.M2["Annualized_Change"].iloc[i] + b

    def plot_score(self, date):
        plt.figure(figsize=[20, 8])
        plt.plot(
            self.PMI["Score"][date:], color="red", linewidth=3, label="PMI", marker="o"
        )
        plt.plot(
            self.M2["Score"][date:], color="black", linewidth=3, label="M2", marker="o"
        )
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.axhline(y=5, color="red", linestyle="dashed")
        plt.axhline(y=10, color="red", linestyle="dashed")
        plt.legend()
        plt.title("Score_Cards")

    def generalize_economy(self):
        self.PMI_Score()
        self.M2_Stat()
        self.M2_Score()
        self.IR_Score()
        self.CPI_all_Score()
        M2 = self.M2.resample("MS").mean()
        a = [
            self.GDP["Change"],
            self.PMI["PMI"],
            M2["Annualized_Change"],
            self.IR["Data"],
            self.IR["Change"],
            self.cpi_all["Change"],
            self.PMI["Score"],
            M2["Score"],
            self.IR["Score"],
            self.cpi_all["Score"],
            self.DXY_M["Data"],
        ]
        self.General = pd.DataFrame(
            a,
            index=[
                "GDP_Growth",
                "PMI",
                "M2_Change",
                "IR",
                "IR_Change",
                "CPI_all_Change",
                "PMI_Score",
                "M2_Score",
                "IR_Score",
                "CPI_All_Score",
                "DXY",
            ],
        ).T
        self.General["Score"] = (
            self.General["PMI_Score"]
            + self.General["IR_Score"]
            + self.General["M2_Score"]
        ) / 3

    def ir_score(self):
        self.IR["Score"] = np.zeros(len(self.IR.index))
        for i in range(len(self.IR.index)):
            if (self.IR["Change"].iloc[i] > 5) & (self.IR["Change"].iloc[i] < 20):
                a = (10 - 3) / (20 - 5)
                b = 3 - 5 * a
                self.IR["Score"].iloc[i] = a * self.IR["Change"].iloc[i] + b
            if (self.IR["Change"].iloc[i] < -10) & (self.IR["Change"].iloc[i] > -20):
                a = (-7 + 3) / (-20 + 10)
                b = -3 + 10 * a
                self.IR["Score"].iloc[i] = a * self.IR["Change"].iloc[i] + b
            if self.IR["Change"].iloc[i] < -30:
                a = (10 - 8) / (-50 + 30)
                b = 8 + 30 * a
                self.IR["Score"].iloc[i] = a * self.IR["Change"].iloc[i] + b

    def cpi_all_score(self):
        self.Inflation_Stat()
        self.cpi_all["Score"] = np.zeros(len(self.cpi_all.index))
        for i in range(len(self.cpi_all.index)):
            if (self.cpi_all["Change"].iloc[i] > 0) & (
                self.cpi_all["Change"].iloc[i] < self.cpi_all_mean
            ):
                self.cpi_all["Score"].iloc[i] = 0
            if (self.cpi_all["Change"].iloc[i] > self.cpi_all_mean) & (
                self.cpi_all["Change"].iloc[i] < self.cpi_all_m2
            ):
                a = (10 - 0) / (self.cpi_all_m2 - self.cpi_all_mean)
                b = -a * self.cpi_all_mean
                self.cpi_all["Score"].iloc[i] = a * self.cpi_all["Change"].iloc[i] + b
            if self.cpi_all["Change"].iloc[i] > self.cpi_all_m2:
                a = (3 - 10) / (self.cpi_all_m3 - self.cpi_all_m2)
                b = 10 - a * self.cpi_all_m2
                self.cpi_all["Score"].iloc[i] = a * self.cpi_all["Change"].iloc[i] + b
            if (self.cpi_all["Change"].iloc[i] < 0) & (
                self.cpi_all["Change"].iloc[i] > self.cpi_all_m_2
            ):
                a = (-10 - 0) / (self.cpi_all_m_2 - 0)
                b = -10 - a * self.cpi_all_m_2
                self.cpi_all["Score"].iloc[i] = a * self.cpi_all["Change"].iloc[i] + b
            if self.cpi_all["Change"].iloc[i] < self.cpi_all_m_2:
                a = (10 - 6) / (self.cpi_all_m_3 - self.cpi_all_m_2)
                b = 6 - a * self.cpi_all_m_2
                self.cpi_all["Score"].iloc[i] = a * self.cpi_all["Change"].iloc[i] + b

    def cpi_foodless_score(self):
        self.Inflation_Stat()
        self.cpi_foodless["Score"] = np.zeros(len(self.cpi_foodless.index))
        for i in range(len(self.cpi_foodless.index)):
            if (self.cpi_foodless["Change"].iloc[i] > 0) & (
                self.cpi_foodless["Change"].iloc[i] < self.cpi_foodless_mean
            ):
                self.cpi_foodless["Score"].iloc[i] = 0
            if (self.cpi_foodless["Change"].iloc[i] > self.cpi_foodless_mean) & (
                self.cpi_foodless["Change"].iloc[i] < self.cpi_foodless_m2
            ):
                a = (10 - 0) / (self.cpi_foodless_m2 - self.cpi_foodless_mean)
                b = -a * self.cpi_foodless_mean
                self.cpi_foodless["Score"].iloc[i] = (
                    a * self.cpi_foodless["Change"].iloc[i] + b
                )
            if self.cpi_foodless["Change"].iloc[i] > self.cpi_foodless_m2:
                a = (3 - 10) / (self.cpi_foodless_m3 - self.cpi_foodless_m2)
                b = 10 - a * self.cpi_foodless_m2
                self.cpi_foodless["Score"].iloc[i] = (
                    a * self.cpi_foodless["Change"].iloc[i] + b
                )
            if (self.cpi_foodless["Change"].iloc[i] < 0) & (
                self.cpi_foodless["Change"].iloc[i] > self.cpi_foodless_m_2
            ):
                a = (-10 - 0) / (self.cpi_foodless_m_2 - 0)
                b = -10 - a * self.cpi_foodless_m_2
                self.cpi_foodless["Score"].iloc[i] = (
                    a * self.cpi_foodless["Change"].iloc[i] + b
                )
            if self.cpi_foodless["Change"].iloc[i] < self.cpi_foodless_m_2:
                a = (10 - 6) / (self.cpi_foodless_m_3 - self.cpi_foodless_m_2)
                b = 6 - a * self.cpi_foodless_m_2
                self.cpi_foodless["Score"].iloc[i] = (
                    a * self.cpi_foodless["Change"].iloc[i] + b
                )

    def read_index(self):
        ########## Read pmi data #############
        service_pmi = pd.read_excel(
            f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="service pmi"
        )
        try:
            for i in service_pmi.index:
                if type(service_pmi.loc[i, "date"]) == str:
                    service_pmi.loc[i, "date"] = pd.to_datetime(
                        service_pmi.loc[i, "date"][:10]
                    )
            service_pmi.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(service_pmi["actual"])
            service_pmi["cycle"] = cycle
            service_pmi["trend"] = trend
        except:
            print(f"add service_pmi {self.country}")
        try:
            manufacturing_pmi = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx",
                sheet_name="manufacturing pmi",
            )
            for i in manufacturing_pmi.index:
                if type(manufacturing_pmi.loc[i, "date"]) == str:
                    manufacturing_pmi.loc[i, "date"] = pd.to_datetime(
                        manufacturing_pmi.loc[i, "date"][:10]
                    )
            manufacturing_pmi.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(manufacturing_pmi["actual"])
            manufacturing_pmi["cycle"] = cycle
            manufacturing_pmi["trend"] = trend
        except:
            print(f"add manufacturing pmi {self.country}")
        ########### Read I.R data #############
        try:
            ir = pd.read_excel(f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="ir")
            # chnge type of data to datetime
            for i in ir.index:
                if type(ir.loc[i, "date"]) == str:
                    ir.loc[i, "date"] = pd.to_datetime(ir.loc[i, "date"][:10])
            ir.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(ir["actual"])
            ir["cycle"] = cycle
            ir["trend"] = trend
        except:
            print(f"add  interest rate {self.country}")
        ############ Read GDP ############
        # GDP Q o Q
        try:
            gdp_q = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="gdp qoq"
            )
            # chnge type of data to datetime
            for i in gdp_q.index:
                if type(gdp_q.loc[i, "date"]) == str:
                    gdp_q.loc[i, "date"] = pd.to_datetime(gdp_q.loc[i, "date"][:10])
            gdp_q.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(gdp_q["actual"])
            gdp_q["cycle"] = cycle
            gdp_q["trend"] = trend
            # normalize_gdp
            for i in np.linspace(2, 40, 60):
                prob_p = len(gdp_q[gdp_q["actual"] > i]) / len(gdp_q)
                if prob_p < 0.02:
                    norm_p = i
                    self.norm_p = norm_p
                    break
            for i in np.linspace(0, -40, 60):
                prob_n = len(gdp_q[gdp_q["actual"] < i]) / len(gdp_q)
                if prob_n < 0.02:
                    norm_n = i
                    self.norm_n = norm_n
                    break
            gdp_q_norm = gdp_q[(gdp_q["actual"] < norm_p) & (gdp_q["actual"] > norm_n)]
            self.gdp_q_norm = gdp_q_norm

        except:
            print(f"add GDP QOQ {self.country}")
        # GDP Y o Y
        try:
            gdp_y = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="gdp yoy"
            )
            # chnge type of data to datetime
            for i in gdp_y.index:
                if type(gdp_y.loc[i, "date"]) == str:
                    gdp_y.loc[i, "date"] = pd.to_datetime(gdp_y.loc[i, "date"][:10])
            gdp_y.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(gdp_y["actual"])
            gdp_y["cycle"] = cycle
            gdp_y["trend"] = trend
        except:
            print(f"add GDP YoY {self.country}")
        ########### Read unemployment_rate ############
        try:
            unemployment = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx",
                sheet_name="unemployment rate",
            )
            # chnge type of data to datetime
            for i in unemployment.index:
                if type(unemployment.loc[i, "date"]) == str:
                    unemployment.loc[i, "date"] = pd.to_datetime(
                        unemployment.loc[i, "date"][:10]
                    )
            unemployment.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(unemployment["actual"])
            unemployment["cycle"] = cycle
            unemployment["trend"] = trend
        except:
            print(f"add unemployment rate {self.country}")
        ########## Read inflation rate #################
        # cpi_mom
        try:
            cpi_m = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="cpi mom"
            )
            # chnge type of data to datetime
            for i in cpi_m.index:
                if type(cpi_m.loc[i, "date"]) == str:
                    cpi_m.loc[i, "date"] = pd.to_datetime(cpi_m.loc[i, "date"][:10])
            cpi_m.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(cpi_m["actual"])
            cpi_m["cycle"] = cycle
            cpi_m["trend"] = trend
        except:
            print(f"add cpi mom {self.country}")
        # cpi_yoy
        try:
            cpi_y = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="cpi yoy"
            )
            # chnge type of data to datetime
            for i in cpi_y.index:
                if type(cpi_y.loc[i, "date"]) == str:
                    cpi_y.loc[i, "date"] = pd.to_datetime(cpi_y.loc[i, "date"][:10])
            cpi_y.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(cpi_y["actual"])
            cpi_y["cycle"] = cycle
            cpi_y["trend"] = trend
        except:
            print(f"add cpi YoY {self.country}")
        # ppi_mom
        try:
            ppi_m = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="ppi mom"
            )
            # chnge type of data to datetime
            for i in ppi_m.index:
                if type(ppi_m.loc[i, "date"]) == str:
                    ppi_m.loc[i, "date"] = pd.to_datetime(ppi_m.loc[i, "date"][:10])
            ppi_m.set_index("date", inplace=True)
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(ppi_m["actual"])
            ppi_m["cycle"] = cycle
            ppi_m["trend"] = trend
        except:
            print(f"add ppi mom {self.country}")
        # ppi_yoy
        try:
            ppi_y = pd.read_excel(
                f"{FOREX_PATH}/{self.country}.xlsx", sheet_name="ppi yoy"
            )
            # chnge type of data to datetime
            for i in ppi_y.index:
                if type(ppi_y.loc[i, "date"]) == str:
                    ppi_y.loc[i, "date"] = pd.to_datetime(ppi_y.loc[i, "date"][:10])
            ppi_y.set_index("date", inplace=True)
        except:
            print(f"add ppi YoY {self.country}")
            # Decompose Cycle and trend
            cycle, trend = hp_filter.hpfilter(ppi_y["actual"])
            ppi_y["cycle"] = cycle
            ppi_y["trend"] = trend
        ########## integrate_data #################
        try:
            data = pd.DataFrame(columns=["unemployment"])
            data["unemployment"] = unemployment["actual"].resample("M").mean()
            data["cpi"] = cpi_y["actual"].resample("M").mean()
            data["ppi"] = ppi_y["actual"].resample("M").mean()
            data["ir"] = ir["actual"].resample("M").mean()
            data["gdp"] = gdp_q["actual"].resample("M").mean()
            data["manu_pmi"] = manufacturing_pmi["actual"].resample("M").mean()
            data.fillna(method="ffill", inplace=True)
            data.dropna(inplace=True)
            self.all_data = data
        except:
            print(f"cant integrate data of {self.country}")
        ########### send data to self ############
        self.service_pmi = service_pmi
        self.manufacturing_pmi = manufacturing_pmi
        self.ir = ir
        self.gdp_q = gdp_q
        self.gdp_y = gdp_y
        self.unemployment = unemployment
        self.cpi_m = cpi_m
        self.ppi_m = ppi_m
        self.cpi_y = cpi_y
        self.ppi_m = ppi_m
        self.ppi_y = ppi_y

    def plot_economy(self):
        # plot pmi and gdp
        ###### Real_index_plot ############
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 3, 1)
        sns.distplot(self.manufacturing_pmi["actual"])
        plt.axvline(self.manufacturing_pmi["actual"].iloc[0], color="red")
        plt.axvline(
            self.manufacturing_pmi["actual"].median(), color="black", linestyle="dashed"
        )
        plt.title("Manufaturing_pmi")
        plt.subplot(1, 3, 2)
        sns.distplot(self.gdp_q_norm["actual"])
        plt.axvline(self.gdp_q["actual"].iloc[0], color="red")
        plt.axvline(self.gdp_q["actual"].median(), color="black", linestyle="dashed")
        plt.title("GDP_Quarterly")
        plt.subplot(1, 3, 3)
        sns.distplot(self.unemployment["actual"])
        plt.axvline(self.unemployment["actual"].iloc[0], color="red")
        plt.axvline(self.unemployment["actual"].median(), linestyle="dashed")
        plt.title("unemployment_rate")
        plt.figure(figsize=[20, 8])
        ########## Nominal_index_plot #############
        plt.subplot(1, 3, 1)
        sns.distplot(self.cpi_y["actual"])
        plt.axvline(self.cpi_y.iloc[0]["actual"], color="red")
        plt.axvline(self.cpi_y["actual"].median(), color="black", linestyle="dashed")
        plt.title("CPI_YoY")
        plt.subplot(1, 3, 2)
        sns.distplot(self.ppi_y["actual"])
        plt.axvline(self.ppi_y.iloc[0]["actual"], color="red")
        plt.axvline(self.ppi_y["actual"].median(), color="black", linestyle="dashed")
        plt.title("PPI_YoY")
        plt.subplot(1, 3, 3)
        sns.distplot(self.ir["actual"])
        plt.axvline(self.ir["actual"].iloc[0], color="red")
        plt.axvline(self.ir["actual"].mean(), color="black", linestyle="dashed")
        plt.title("I.R")
