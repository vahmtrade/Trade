import os
import re
import random
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytse_client as tse
import sklearn.linear_model as linear
import sklearn.metrics as met
import seaborn as sns
import arabic_reshaper
import tse_index
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
import itertools
from cmath import tan
from itertools import cycle
from scipy import stats
from persiantools.jdatetime import JalaliDate
from statsmodels.tsa.filters import hp_filter
from bidi.algorithm import get_display
from sklearn.metrics import mean_squared_error
from statics.setting import *
from preprocess.basic import *
from process.strategy import *
import finpy_tse as fpy
import warnings
from sklearn.metrics import r2_score, mean_absolute_percentage_error, accuracy_score
import plotly.data as pdd
import plotly.express as px
import pmdarima as pm

warnings.filterwarnings("ignore")
plt.style.use("seaborn")


class IncomeDataFrame(pd.DataFrame):
    def __init__(
        self,
        data=None,
        dependent_df=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
    ):
        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.dependent_df = dependent_df
        self.update_dependent_columns()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "فروش" or key == "Cost_of_Revenue":
            self.update_dependent_columns()

    def update_dependent_columns(self):
        if self.dependent_df is not None:
            self.loc[:, "بهای تمام شده کالای فروش رفته"] = -self.dependent_df[
                "جمع بهای تمام شده"
            ]
        self.loc[:, "سود (زیان) ناخالص"] = (
            self["فروش"] + self["بهای تمام شده کالای فروش رفته"]
        )
        self.loc[:, "سود (زیان) عملیاتی"] = (
            self["سود (زیان) ناخالص"]
            + self["هزینه های عمومی, اداری و تشکیلاتی"]
            + self["خالص سایر درامدها (هزینه ها) ی عملیاتی"]
        )
        self.loc[:, "سود (زیان) خالص عملیات در حال تداوم قبل از مالیات"] = (
            self["سود (زیان) عملیاتی"]
            + self["هزینه های مالی"]
            + self["خالص سایر درامدها و هزینه های غیرعملیاتی"]
        )
        self.loc[:, "سود (زیان) خالص عملیات در حال تداوم"] = (
            self["سود (زیان) خالص عملیات در حال تداوم قبل از مالیات"] + self["مالیات"]
        )
        self.loc[:, "سود (زیان) خالص"] = self.loc[
            :, "سود (زیان) خالص عملیات در حال تداوم"
        ]
        try:
            self.loc[:, "سود هر سهم پس از کسر مالیات"] = (
                self.loc[:, "سود (زیان) خالص"] * 1000 / self["سرمایه"]
            )
        except:
            pass
        try:
            self.loc[:, "سود هر سهم بر اساس آخرین سرمایه"] = (
                self.loc[:, "سود (زیان) خالص"] * 1000 / self["سرمایه"].iloc[-1]
            )
        except:
            pass


class CostDataFrame(pd.DataFrame):
    def __init__(
        self,
        data=None,
        dependent_df=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
    ):
        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.dependent_df = dependent_df
        self.update_dependent_columns()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "فروش" or key == "Cost_of_Revenue":
            self.update_dependent_columns()

    def update_dependent_columns(self):
        if self.dependent_df is not None:
            self.loc[:, "سربار تولید"] = self.dependent_df["جمع"]
        self.loc[:, "جمع"] = (
            self["مواد مستقیم مصرفی"]
            + self["دستمزد مستقیم تولید"]
            + self["سربار تولید"]
        )
        if "هزینه جذب نشده در تولید" in self.columns:
            self.loc[:, "جمع هزینه های تولید"] = (
                self["جمع"] + self["هزینه جذب نشده در تولید"]
            )
        else:
            self.loc[:, "جمع هزینه های تولید"] = self["جمع"]
        if "ضایعات غیرعادی" in self.columns:
            self.loc[:, "بهای تمام شده کالای تولید شده"] = (
                self["جمع هزینه های تولید"] + self["ضایعات غیرعادی"]
            )
        else:
            self.loc[:, "بهای تمام شده کالای تولید شده"] = self["جمع هزینه های تولید"]
        self.loc[:, "بهای تمام شده کالای فروش رفته"] = (
            self["بهای تمام شده کالای تولید شده"]
            + self["موجودی کالای ساخته شده اول دوره"]
            + self["موجودی کالای ساخته شده پایان دوره"]
        )
        if "بهای تمام شده خدمات ارایه شده" in self.columns:
            self.loc[:, "جمع بهای تمام شده"] = (
                self["بهای تمام شده کالای فروش رفته"]
                + self["بهای تمام شده خدمات ارایه شده"]
            )
        else:
            self.loc[:, "جمع بهای تمام شده"] = self["بهای تمام شده کالای فروش رفته"]


class OverheadDataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.update_dependent_columns()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "فروش" or key == "Cost_of_Revenue":
            self.update_dependent_columns()

    def update_dependent_columns(self):
        self.loc[:, "جمع"] = self.sum(axis=1) - self["جمع"]


def convert_monthly_dataframe(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    ys = int(df.index[0].split("-")[0])
    ms = int(df.index[0].split("-")[1])
    ds = int(df.index[0].split("-")[2])
    ye = int(df.index[-1].split("-")[0])
    me = int(df.index[-1].split("-")[1])
    de = int(df.index[-1].split("-")[2])
    months = (ye - ys) * 12 + (me - ms) + 1
    mstart = ms
    ystart = ys
    m = mstart
    y = ystart
    prices = []
    dates = []
    for i in range(months):
        if m < 10:
            try:
                month_price = df.loc[f"{y}-0{m}-01":f"{y}-0{m}-30"].iloc[-1].values[0]
            except:
                month_price = prices[-1]
        else:
            try:
                month_price = df.loc[f"{y}-{m}-01":f"{y}-{m}-30"].iloc[-1].values[0]
            except:
                month_price = prices[-1]

        date = f"{y}/{m}"
        prices.append(month_price)
        dates.append(date)
        m = m + 1
        if m > 12:
            m = 1
            y = y + 1
    monthly_df = pd.DataFrame(columns=["Close"], index=dates)
    monthly_df["Close"] = prices
    monthly_df["Ret"] = monthly_df["Close"].pct_change()
    monthly_df.dropna(inplace=True)
    return monthly_df


def fill_outlier_data(df, alpha=10):
    for i in df.columns:
        iqr = df[i].quantile(0.75) - df[i].quantile(0.25)
        up_threshold = df[i].quantile(0.75) + alpha * iqr
        down_threshold = df[i].quantile(0.25) - alpha * iqr
        df[i] = df[i].apply(
            lambda x: df[i].median()
            if ((x > up_threshold) or (x < down_threshold))
            else x
        )


def walkforward(
    df,
    h,
    steps,
    trend_type,
    seasonal_type,
    damped_trend,
    init_method,
    use_boxcox,
    debug=False,
):
    errors = []
    seen_last = False
    steps_completed = 0
    ntest = len(df) - h - steps + 1
    for end_train in range(ntest, len(df) - h + 1):
        train = df.iloc[:end_train]
        test = df.iloc[end_train : end_train + h]
        if test.index[-1] == df.index[-1]:
            seen_last = True
        steps_completed += 1
        hw = ExponentialSmoothing(
            train,
            trend=trend_type,
            seasonal=seasonal_type,
            seasonal_periods=12,
            initialization_method=init_method,
            use_boxcox=use_boxcox,
            damped_trend=damped_trend,
        )
        res_hw = hw.fit()
        fcast = res_hw.forecast(h)
        try:
            error = mean_squared_error(test, fcast)
        except:
            break
        errors.append(error)
        model = ExponentialSmoothing(
            df,
            trend=trend_type,
            seasonal=seasonal_type,
            seasonal_periods=12,
            initialization_method=init_method,
            use_boxcox=use_boxcox,
            damped_trend=damped_trend,
        )
        res_model = model.fit()
    if debug:
        print("seen_last :", seen_last)
        print("steps completed", steps_completed)
    return np.mean(errors), res_model, seen_last


def best_hw_model(df, h, steps):
    trend_type_lst = ["add", "mul"]
    seasonal_type_lst = ["add", "mul"]
    damped_trend_lst = [True, False]
    init_method_lst = ["estimated", "heuristic", "legacy-heuristic"]
    use_box_cox_lst = [True, False]
    tuple_option_lists = (
        trend_type_lst,
        seasonal_type_lst,
        damped_trend_lst,
        init_method_lst,
        use_box_cox_lst,
    )
    best_score = float("inf")
    best_option = None
    for x in itertools.product(*tuple_option_lists):
        score = walkforward(df, h, steps, *x)[0]
        seen_last = walkforward(df, h, steps, *x)[2]
        if (score < best_score) and (seen_last == True):
            print("best_score", score)
            best_score = score
            best_option = x
            best_model = walkforward(df, h, steps, *x)[1]
    return best_option, best_model


def get_coloumn(df, fiscal_year):
    fiscal_dic = {
        12: {3: 1, 6: 2, 9: 3, 12: 4},
        9: {12: 1, 3: 2, 6: 3, 9: 4},
        6: {9: 1, 12: 2, 3: 3, 6: 4},
        3: {6: 1, 9: 2, 12: 3, 3: 4},
        10: {1: 1, 4: 2, 7: 3, 10: 4},
        8: {11: 1, 2: 2, 5: 3, 8: 4},
    }

    all_time_id = re.findall(regex_en_timeid_q, str(df))
    n = len(all_time_id)

    my_month = int(all_time_id[-1][5:])
    my_Q = fiscal_dic[fiscal_year][my_month]
    if my_Q == 4:
        my_year = my_year - 1

    for i in df.iloc[0][1:]:
        i = i[0:10]
        b = [int(j) for j in i.split("-")]
        year = b[0]
        month = b[1]
        day = b[2]
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
    return my_col


def best_stepwise_reg(x, y):
    while len(x.columns) > 0:
        x_subset = sm.add_constant(x)
        model = sm.OLS(y, x_subset).fit()
        pval = model.pvalues
        pval_max = pval.max()
        var_max = pval.idxmax()
        if pval_max > 0.1:
            x.drop(var_max, axis=1, inplace=True)
        else:
            best_model = model
            break
    return best_model


def save_watchlist(date=today_8char):
    """save all stocks data in one file"""

    data = {}
    errs = {}

    for stock in wl_prod:
        try:
            data[stock] = Stock(stock)

        except Exception as err:
            errs[stock] = err

    with open(f"{PKLPATH}/{date}.pkl", "wb") as file:
        pickle.dump(data, file)

    return errs


def read_watchlist(date):
    with open(f"{PKLPATH}/{date}.pkl", "rb") as f:
        data = pickle.load(f)

    return data


def analyse_watchlist(date):
    data = read_watchlist(date)

    analyse = pd.DataFrame(
        index=data.keys(),
        columns=[
            "sector",
            "margin",
            "margin_var",
            "last_gross_margin",
            "gross_margin_change",
            "last_rate/mean_12",
            "last_rate_change",
            "last_rate/same_year",
            "سود (زیان) خالص_change",
            "سود (زیان) خالص/سود (زیان) خالص_same_last",
        ],
    )
    errs = {}
    for stock in data.values():
        try:
            df = stock.price_revenue_com_yearly.copy()
            df.drop(["total", "جمع"], axis=1, inplace=True)
            converte_numeric(df)
            max_com = df.idxmax(axis=1).iloc[-1]
            analyse.loc[stock.Name]["sector"] = stock.industry
            analyse.loc[stock.Name]["margin"] = stock.Risk_pred_incomeearly[0]
            analyse.loc[stock.Name]["margin_var"] = stock.Risk_income_yearly[1]
            analyse.loc[stock.Name][
                "last_gross_margin"
            ] = stock.income_common_rial_quarterly["Gross_Profit"].iloc[-1]
            analyse.loc[stock.Name]["gross_margin_change"] = (
                stock.income_common_rial_quarterly["Gross_Profit"].pct_change().iloc[-1]
            )
            analyse.loc[stock.Name]["last_rate/mean_12"] = (
                stock.rate_monthly[max_com].iloc[-1]
                / stock.rate_monthly[max_com].iloc[-12:].mean()
                - 1
            )
            analyse.loc[stock.Name]["last_rate_change"] = stock.rate_change_monthly[
                max_com
            ].iloc[-1]
            analyse.loc[stock.Name]["last_rate/same_year"] = (
                stock.rate_monthly.iloc[-1][max_com]
                / stock.rate_monthly.iloc[-13][max_com]
                - 1
            )
            analyse.loc[stock.Name]["سود (زیان) خالص_change"] = (
                stock.income_quarterly["سود (زیان) خالص"].pct_change().iloc[-1]
            )
            analyse.loc[stock.Name]["سود (زیان) خالص/سود (زیان) خالص_same_last"] = (
                stock.income_quarterly["سود (زیان) خالص"].iloc[-1]
                / stock.income_quarterly["سود (زیان) خالص"].iloc[-5]
                - 1
            )

        except Exception as err:
            errs[stock.Name] = err
        technical = pd.DataFrame(
            columns=["sector", "price", "sig_buy", "sig_sell"], index=data.keys()
        )
    for stock in data.values():
        try:
            technical.loc[stock.Name]["sector"] = stock.industry
            technical.loc[stock.Name]["price"] = stock.Price["Close"].iloc[-1]
            technical.loc[stock.Name]["sig_buy"] = stock.my_tester.trade[
                "Sig_Price_Buy"
            ].iloc[-1]
            technical.loc[stock.Name]["sig_sell"] = stock.my_tester.trade[
                "Sig_Price_Sell"
            ].iloc[-1]
        except Exception as err:
            errs[stock.Name] = err
    value = pd.DataFrame(
        columns=[
            "sector",
            "price",
            "value",
            "pe_terminal_historical",
            "pe_terminal_capm",
            "pe_terminal",
            "pe_fw",
            "g_stock",
            "g_longterm",
            "expected_return_historical",
            "expected_return_capm",
            "expected_return",
            "beta",
            "cagr_count",
        ],
        index=data.keys(),
    )
    for stock in data.values():
        try:
            value.loc[stock.Name]["sector"] = stock.industry
            value.loc[stock.Name]["price"] = stock.Price["Close"].iloc[-1]
            value.loc[stock.Name]["value"] = stock.value
            value.loc[stock.Name][
                "pe_terminal_historical"
            ] = stock.pe_terminal_historical
            value.loc[stock.Name]["pe_terminal_capm"] = stock.pe_terminal_capm
            value.loc[stock.Name]["pe_terminal"] = stock.pe_terminal
            value.loc[stock.Name]["pe_fw"] = stock.pe_fw
            value.loc[stock.Name]["g_stock"] = stock.g_stock
            value.loc[stock.Name]["g_longterm"] = stock.g_economy + 1
            value.loc[stock.Name]["expected_return_historical"] = stock.k_historical
            value.loc[stock.Name]["expected_return_capm"] = stock.k_capm
            value.loc[stock.Name]["expected_return"] = stock.k
            value.loc[stock.Name]["beta"] = stock.beta
            value.loc[stock.Name]["cagr_count"] = stock.cagr_count - 1
        except Exception as err:
            errs[stock.Name] = err
    try:
        value.drop("deshimi", inplace=True)
    except:
        pass
    value["value/price"] = value["value"] / value["price"]
    d = {
        "analyse": analyse,
        "technical": technical,
        "value": value,
        "data": data,
        "err": errs,
    }
    value_price = value[["value/price"]].applymap(lambda x: 0 if x < 0 else x)
    group_analyse = analyse.groupby("sector").median()
    try:
        group_analyse.drop("palayesh", inplace=True)
    except:
        pass
    group_value = value.groupby("sector").median()
    plt.figure(figsize=[20, 8])
    plt.subplot(1, 3, 1)
    group_analyse["last_rate_change"].plot(kind="bar")
    plt.title("rate/last_rate")
    plt.subplot(1, 3, 2)
    group_analyse["last_rate/same_year"].plot(kind="bar")
    plt.title("last_rate/same_year")
    plt.subplot(1, 3, 3)
    group_analyse["last_rate/mean_12"].plot(kind="bar")
    plt.title("last_rate/mean_12")
    plt.figure(figsize=[20, 8])
    plt.subplot(1, 2, 1)
    group_analyse["سود (زیان) خالص_change"].plot(kind="bar")
    plt.title("سود (زیان) خالص/سود (زیان) خالص_last")
    plt.subplot(1, 2, 2)
    group_analyse["سود (زیان) خالص/سود (زیان) خالص_same_last"].plot(kind="bar")
    plt.title("سود (زیان) خالص/سود (زیان) خالص_same_last")
    value_price.plot(kind="bar", figsize=[20, 8])
    plt.axhline(y=1, linestyle="dashed", color="red")
    return d, [group_analyse, group_value, value_price]


def update_watchlist(data):
    # update price of data
    for stock in data.values():
        stock.Price, stock.Price_dollar = read_stock(
            stock.farsi, stock.start_date, stock.end_date
        )
        data[stock.Name] = stock

    return data


def plot_valuation_history(stock_name):
    x, y = [], []
    for i in os.listdir(PKLPATH):
        try:
            date = i.split(".")[0]
            data = read_watchlist(date)
            y.append(data[stock_name].value)
            x.append(date)
        except:
            pass
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.title(stock_name)
    plt.plot(x, y, marker="o")
    plt.show()


def analyse_detail_trade(adress):
    # read raw data
    df = pd.read_excel(adress)
    # select desired data
    data = df[["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5"]]
    # preprocess data
    data.rename(
        columns={
            "Unnamed: 2": "n",
            "Unnamed: 3": "time",
            "Unnamed: 4": "volume",
            "Unnamed: 5": "price",
        },
        inplace=True,
    )
    data.drop(0, inplace=True)
    data["value"] = (data["price"] * data["volume"]) / 10**7
    # same time
    a = data[data["time"].duplicated(keep=False)]
    # change same time
    b = a["time"].duplicated()
    count = 0
    temp_first = []
    temp_last = []
    for i in range(len(b.index)):
        if b.loc[b.index[i]] == False:
            count += 1
            if count == 1:
                temp_first.append(b.index[i])
            if count == 2:
                temp_last.append(b.index[i - 1])
                count = 0
    while len(temp_first) > len(temp_last):
        temp_first.pop(-1)
    volume = []
    ch_price = []
    value = []
    number = []
    time = []
    price = []
    # fill desired data
    for i in range(len(temp_first)):
        volume.append(a.loc[temp_first[i] : temp_last[i]]["volume"].sum())
        value.append(a.loc[temp_first[i] : temp_last[i]]["value"].sum())
        ch_price.append(a.loc[temp_last[i]]["price"] / a.loc[temp_first[i]]["price"])
        number.append(len(a.loc[temp_first[i] : temp_last[i]]))
        time.append(a.loc[temp_first[i]]["time"])
        price.append(a.loc[temp_first[i]]["price"])
    process_detail = pd.DataFrame(columns=["volume", "price"])
    process_detail["volume"] = volume
    process_detail["price_ch"] = ch_price
    process_detail["value"] = value
    process_detail["number"] = number
    process_detail["time"] = time
    process_detail["price"] = price
    return process_detail


def load_stock_analyse(stock_name, name):
    """
    load your analyse from stock_name/analyse/name.pkl
    """
    indus = wl_prod[stock_name]["indus"]
    with open(f"{INDUSPATH}/{indus}/{stock_name}/analyse/{name}.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def show_analyze(stock_name):
    indus = wl_prod[stock_name]["indus"]
    for i in os.listdir(f"{INDUSPATH}/{indus}/{stock_name}/analyse"):
        print(i)


def random_plot(n):
    number = 200
    chart = []
    for i in range(n):
        number += random.choice([1, -1])
        chart.append(number)
    plt.figure(figsize=[20, 8])
    plt.title("random_chart")
    plt.plot(chart)
    plt.grid()
    return chart


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
    pe = pd.read_excel(
        f"{MACROPATH}/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=1,
    )
    pe.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    pe.set_index("Jalali", inplace=True)
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
        f"{MACROPATH}/macro.xlsx",
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
        f"{MACROPATH}/macro.xlsx",
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
        f"{MACROPATH}/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=4,
    )
    dollar_nima.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    dollar_azad.set_index("Jalali", inplace=True)
    dollar_nima.set_index("Jalali", inplace=True)
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
        f"{MACROPATH}/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=6,
    )
    IR.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Close"},
        inplace=True,
    )
    IR.set_index("Jalali", inplace=True)
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
        f"{MACROPATH}/macro.xlsx",
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
    direct.set_index("Jalali", inplace=True)
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
        f"{MACROPATH}/macro.xlsx",
        skiprows=3,
        parse_dates=["تاریخ میلادی"],
        engine="openpyxl",
        sheet_name=0,
    )
    value.rename(
        columns={"تاریخ میلادی": "Date", "تاریخ شمسی": "Jalali", "مقدار": "Value"},
        inplace=True,
    )
    value.set_index("Jalali", inplace=True)
    value.dropna(inplace=True)
    value = value[start:end]
    value["sma_s"] = value["Value"].rolling(sma_s).mean()
    value["sma_l"] = value["Value"].rolling(sma_l).mean()
    value.dropna(inplace=True)
    return value


def read_value_fix(start, end, sma_s, sma_l):
    value = pd.read_excel(
        f"{MACROPATH}/macro.xlsx",
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
    """
    name:farsi symbol,start_date,end_date:miladi return stock and stock_dollar_price
    """
    stock = fpy.Get_Price_History(
        stock=name, start_date=start_date, end_date=end_date, adjust_price=True
    )
    stock = stock[
        [
            "Adj Open",
            "Adj High",
            "Adj Low",
            "Adj Close",
            "Adj Final",
            "No",
            "Value",
            "Name",
            "Market",
        ]
    ]
    stock.rename(
        columns={
            "Adj Open": "open",
            "Adj High": "high",
            "Adj Low": "low",
            "Adj Close": "Close",
            "Adj Final": "final",
            "No": "number",
            "Value": "Value",
            "Market": "market",
            "Name": "name",
        },
        inplace=True,
    )
    stock["Change"] = stock["Close"].pct_change()
    stock.dropna(inplace=True)
    stock["Ret"] = np.log(stock["Close"] / stock["Close"].shift(1))
    stock["Cret"] = stock["Ret"].cumsum().apply(np.exp)
    stock["Cummax"] = stock["Cret"].cummax()
    stock["Drow_Down"] = stock["Cummax"] - stock["Cret"]
    stock.dropna(inplace=True)
    try:
        dollar_azad, dollar_nima = read_dollar(start_date, end_date)
        stock_dollar = stock["Close"].copy()
        stock_dollar = stock_dollar.to_frame()
        stock_dollar["Close"] = stock_dollar["Close"] / dollar_azad["Close"]
        stock_dollar["Change"] = stock_dollar["Close"].pct_change()
        stock_dollar["Ret"] = np.log(
            stock_dollar["Close"] / stock_dollar["Close"].shift(1)
        )
        stock_dollar["Cret"] = stock_dollar["Ret"].cumsum().apply(np.exp)
        stock_dollar["Cummax"] = stock_dollar["Cret"].cummax()
        stock_dollar["Drow_Down"] = stock_dollar["Cummax"] - stock_dollar["Cret"]
    except:
        stock_dollar = 0
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
    lst = []
    for i in Portfolio["Stock"]:
        try:
            lst.append(wl_prod_df[wl_prod_df["token"] == i].index[0])
        except:
            print(f"no {i}")
    Portfolio["Stock"] = lst
    Portfolio.set_index("Stock", inplace=True)
    data = read_watchlist("1401-10-15")
    Portfolio["pe_fw"] = np.zeros(len(Portfolio))
    for i in Portfolio.index:
        Portfolio["Industry"].loc[i] = wl_prod[i]["indus"]
        try:
            Portfolio["pe_fw"].loc[i] = data[i].pe_fw
        except:
            Portfolio["pe_fw"].loc[i] = np.nan
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
    for i in Portfolio.index:
        try:
            ticker = tse.Ticker(wl_prod[i]["token"])
            a.append(ticker.p_e_ratio)
        except:
            a.append(np.nan)
    Portfolio["PE"] = a
    Portfolio.to_excel(f"{DB}/Portfolio/{broker}/{owner}/Process_data/{date}.xlsx")
    return Portfolio


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


def create_quaretrly_columns(df, fiscal_year):
    fiscal_dic = {
        12: {3: 1, 6: 2, 9: 3, 12: 4},
        9: {12: 1, 3: 2, 6: 3, 9: 4},
        6: {9: 1, 12: 2, 3: 3, 6: 4},
        3: {6: 1, 9: 2, 12: 3, 3: 4},
        10: {1: 1, 4: 2, 7: 3, 10: 4},
        8: {11: 1, 2: 2, 5: 3, 8: 4},
    }

    all_time_id = re.findall(regex_en_timeid_q, str(df.loc[6]))
    n = len(all_time_id)
    my_month = int(all_time_id[-1][5:])
    my_Q = fiscal_dic[12][my_month]
    my_year = int(all_time_id[-1][:4])
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
    return my_col


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


def plot_marq(stocks, y_s=1400, m_s=1, d_s=1, y_e=1401, m_e=12, d_e=1):
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
        margin_mean_quarterly.append(i.Risk_income_quarterly[0])
        margin_risk_quarterly.append(i.Risk_income_quarterly[1])
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
        plt.plot(i.dollar_analyse["فروش"], label=i.Name, marker="o")
    plt.legend()
    plt.title("Total_Rev_Dollar")
    plt.figure(figsize=[16, 8])
    for i in stocks:
        plt.plot(i.dollar_analyse["سود (زیان) خالص"], label=i.Name, marker="o")
    plt.legend()
    plt.title("سود (زیان) خالص_Dollar")


def plot_stocks_ret(
    stocks, year_s=1401, month_s=1, day_s=1, year_e=1401, month_e=12, day_e=1
):
    start = pd.to_datetime(JalaliDate(year_s, month_s, day_s).to_gregorian())
    end = pd.to_datetime(JalaliDate(year_e, month_e, day_e).to_gregorian())
    plt.figure(figsize=[20, 8])
    for i in stocks:
        plt.plot(
            i.Price[start:end]["Close"] / i.Price[start:end]["Close"].iloc[0],
            label=i.Name,
        )
        plt.legend()
    plt.figure(figsize=[20, 8])
    for i in stocks:
        plt.bar(
            x=i.Name,
            height=i.Price[start:end]["Close"].iloc[-1]
            / i.Price[start:end]["Close"].iloc[0],
        )


def plot_param_stocks(stocks, parametr):
    """
    parametr is: pe_terminal,pe_fw,growth,count,rate,pe_capm,pe_terminal_historical,'k_capm','k_historical','k','beta,pot
    """
    if parametr == "pe_terminal":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.pe_terminal)
            except:
                pass
    if parametr == "pe_fw":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.pe_fw)
            except:
                pass
    if parametr == "growth":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.g_stock)
            except:
                pass
    if parametr == "count":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.cagr_count)
            except:
                pass
    if parametr == "rate":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.cagr_rate)
            except:
                pass
    if parametr == "pe_capm":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.pe_terminal_capm)
            except:
                pass
    if parametr == "pe_terminal_historical":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.pe_terminal_historical)
            except:
                pass
    if parametr == "k_capm":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.k_capm)
            except:
                pass
    if parametr == "k_historical":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.k_historical)
            except:
                pass
    if parametr == "k":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.k)
            except:
                pass
    if parametr == "beta":
        plt.figure(figsize=[12, 6])
        for i in stocks:
            try:
                plt.bar(x=i.Name, height=i.beta)
            except:
                pass
    if parametr == "pot":
        plt.figure(figsize=[12, 6])
        plt.axhline(y=1, linestyle="dashed")
        for i in stocks:
            try:
                pot = i.value / i.Price["Close"].iloc[-1]
                if pot < 0:
                    pot = 0
                plt.bar(x=i.Name, height=pot)
            except:
                pass


def plot_pe_trend_stocks(stocks, year_s=1400, month_s=1, year_e=1401, month_e=12):
    start = pd.to_datetime(JalaliDate(year_s, month_s, 1).to_gregorian())
    end = pd.to_datetime(JalaliDate(year_e, month_e, 1).to_gregorian())
    plt.figure(figsize=[20, 8])
    for i in stocks:
        plt.plot(i.pe[end:start]["P/E-ttm"], label=i.Name)
    plt.legend()


def plot_revenue_stocks(stocks):
    plt.figure(figsize=[20, 8])


def plot_corr(stocks, y_s=1401, m_s=1, y_e=1401, m_e=12):
    d = {}
    date_1 = pd.to_datetime(JalaliDate(y_s, m_s, 1).to_gregorian())
    date_2 = pd.to_datetime(JalaliDate(y_e, m_e, 1).to_gregorian())
    for i in stocks:
        d[i.Name] = i.Price[date_1:date_2]["Ret"]
    df = pd.DataFrame(d)
    df.fillna(0, inplace=True)
    corr = df.corr()
    # plt.figure(figsize=[20, 12], facecolor="white")
    # sns.set(font_scale=1.5)
    # sns.heatmap(cor, cmap="Reds", annot=True, annot_kws={"size": 12}, vmax=1, vmin=-1)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(
            corr, mask=mask, vmax=0.4, square=True, annot=True, cmap="RdYlGn"
        )
    return df


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


def get_pe_data(name):
    adress = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['pe']}"
    pe = pd.read_excel(
        adress,
        engine="openpyxl",
        usecols="B,C,S,R,Q",
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


def select_df(df, str1, str2):
    first = []
    end = []
    ########search for str1 str2##########
    for i in df.index:
        for j in df.columns:
            if isinstance(df.loc[i, j], str):
                if str1 in df.loc[i, j]:
                    first.append(i)

            if isinstance(df.loc[i, j], str):
                if df.loc[i, j] == str2:
                    end.append(i)
    a = []
    #########search for str2 subsequent str1##########
    for i in end:
        if i - first[0] > 0:
            a.append(i)

    resault = df.loc[first[0] : a[0]]
    ############preprocess resault#############
    resault.dropna(axis=1, how="all", inplace=True)
    resault.dropna(axis=0, how="all", inplace=True)
    # remove '-' from data
    for i in resault.index:
        for j in resault.columns:
            if resault.loc[i, j] == "-":
                resault.loc[i, j] = 0
    return resault


def delete_empty(df):
    c = 0
    empty = []
    for i in df.columns:
        for j in df[i]:
            if (j == 0.01) | (j == 0):
                c += 1
        if c == len(df):
            empty.append(i)
        c = 0
    df.drop(empty, axis=1, inplace=True)


def remove_zero(df):
    values_to_check = [0, 0.01]

    # Drop columns where all values are in the list of values_to_check
    df = df.loc[:, ~(df.isin(values_to_check)).all()]
    df = df.loc[~(df.isin(values_to_check)).all(axis=1)]
    return df


def search_df_month(df, fiscal_year, future_year, word):
    try:
        count = 0
        count_last = 0
        m = []
        # search in monthly product
        for i in df.index:
            if (fiscal_year == 12) & (int(i.split("/")[0]) == future_year):
                count += df.loc[i][word]
                m.append(i[5:7])
            if (fiscal_year != 12) & (
                (int(i.split("/")[0]) == future_year)
                | (
                    (int(i.split("/")[0]) == future_year - 1)
                    & (int(i.split("/")[1]) > fiscal_year)
                )
            ):
                count += df.loc[i][word]
                # month done from fiscal_year
                m.append(i)

        if fiscal_year == 12:
            last_index = [f"1400/{i}" for i in m]
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
                s = f"{l[i]}/{month[i]}"
                last_index.append(s)
        for i in last_index:
            count_last += df.loc[i][word]
    except:
        count = 0
        count_last = 0
    return [count_last, count]


def drop_non_same_columns(df1, df2):
    df1_ex = []
    df2_ex = []
    for i in df1.columns:
        if i not in df2.columns:
            df1_ex.append(i)
    df1.drop(df1_ex, axis=1, inplace=True)
    for i in df2.columns:
        if i not in df1.columns:
            df2_ex.append(i)
    df2.drop(df2_ex, axis=1, inplace=True)


def drop_non_same_index(df1, df2):
    df1_ex = []
    df2_ex = []
    for i in df1.index:
        if i not in df2.index:
            df1_ex.append(i)
    df1.drop(df1_ex, inplace=True)
    for i in df2.index:
        if i not in df1.index:
            df2_ex.append(i)
    df2.drop(df2_ex, inplace=True)


def drop_non_same_columns_1(df1, df2):
    df1_ex = []
    df2_ex = []
    for i in df1.columns:
        if i not in df2.columns:
            df1_ex.append(i)
    df1.drop(df1_ex, axis=1, inplace=True)


def find_rate(df, word):
    try:
        i = 0
        while df[word].iloc[-1 - i] == 1:
            i += 1
        rate = df[word].iloc[-1 - i]
    except:
        rate = 0
    return rate


def search_df_quarterly(df, future_year, word):
    count = 0
    q = 0
    try:
        for i in df.index:
            if int(i[:4]) == future_year:
                count += df.loc[i][word]
                q = int(i[-1])
    except:
        count = 0
        q = 0
    return [count, q]


def merge_similar_columns(df):
    df_c = df.copy()

    for i in df_c.columns:
        a = df.columns
        a = a.drop(i)
        for j in a:
            if i in j:
                try:
                    df_c[i] += df_c[j]
                    df_c.drop(j, axis=1, inplace=True)
                except:
                    pass
    return df_c


def rename_columns_dfs(df1, df2):
    lst_min = []
    lst_max = []
    for i in df1.columns:
        for j in df2.columns:
            if i in j or j in i:
                i, j = min(i, j), max(i, j)
                lst_min.append(i)
                lst_max.append(j)
    dic = dict(zip(lst_max, lst_min))
    try:
        df1.rename(columns=dic, inplace=True)
    except:
        print("1 error")

    try:
        df2.rename(columns=dic, inplace=True)
    except:
        print("2 error")

    return dic


def add_extra_columns_dfs(df1, df2):
    for i in df1.columns:
        if i not in df2.columns:
            df2[i] = 0
    for i in df2.columns:
        if i not in df1.columns:
            df1[i] = 0


def merge_same_columns(df):
    dup_c = df.columns[df.columns.duplicated()]
    for i in dup_c:
        try:
            df["temp"] = df[i].sum(axis=1)
            df.drop(i, axis=1, inplace=True)
        except:
            pass
        try:
            df.rename(columns={"temp": i}, inplace=True)
        except:
            pass
    return dup_c


def merge_same_index(df):
    dup_c = df.index[df.index.duplicated()]
    for i in dup_c:
        try:
            df.loc["temp"] = df.loc[i].sum(axis=0)
            df.drop(i, axis=0, inplace=True)
        except:
            pass
        try:
            df.rename(index={"temp": i}, inplace=True)
        except:
            pass
    return dup_c


def converte_numeric(df):
    for i in df.columns:
        try:
            df[i] = pd.to_numeric(df[i])
        except:
            pass


def monte_carlo_simulate(data, num_simulations=100, num_days=365):
    prices = data.tolist()

    # Calculate daily returns
    returns = data.pct_change().dropna().tolist()

    last_price = prices[-1]

    # Set up simulation
    simulation_df = pd.DataFrame()
    for i in range(num_simulations):
        count = 0
        daily_volatility = np.std(returns)
        price_series = []
        price_series.append(last_price)

        # Run simulation for each day
        for d in range(num_days):
            daily_returns = np.random.normal(np.mean(returns), daily_volatility)
            price = price_series[count] * (1 + daily_returns)
            price_series.append(price)
            count += 1

        simulation_df[i] = price_series

    # Plot simulations
    # fig = plt.figure(figsize=(16, 8))
    # plt.plot(simulation_df)
    # plt.xlabel("Day")
    # plt.ylabel("Price")
    # plt.title("Monte Carlo Simulation for Stock Prices")
    # plt.show()
    return simulation_df


def create_month_id(month, year):
    if month < 10:
        id = f"{year}/0{month}"
    if month >= 10:
        id = f"{year}/{month}"
    return id


def replace_negative_data(df):
    # iterate over each column of the DataFrame
    for col in df.columns:
        # check if any value in the column is negative
        if df[col].lt(0).any():
            # calculate the median of the column
            median = df[col].median()
            # replace negative values with the median
            df.loc[df[col] < 0, col] = median


def convert_to_miladi(str):
    str_spl = str.split("-")
    dates = [int(x) for x in str_spl]
    return pd.to_datetime(JalaliDate(dates[0], dates[1], dates[2]).to_gregorian())


def convert_to_jalali(date):
    return f"{JalaliDate.to_jalali(date).year}-{JalaliDate.to_jalali(date).month:02}-{JalaliDate.to_jalali(date).day:02}"


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


class Macro:
    def __init__(
        self,
        start="1391-01-01",
        end="1403-01-01",
        pe_f=6.5,
        sma_s=20,
        sma_l=100,
    ):
        self.start = start
        self.end = end
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.dollar_azad, self.dollar_nima = read_dollar(self.start, self.end)
        self.shakhes_kol = fpy.Get_CWI_History(start_date=self.start, end_date=self.end)
        self.sahkhes_ham = fpy.Get_EWI_History(start_date=self.start, end_date=self.end)
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
        try:
            self.get_historical_data()
        except:
            pass
        # create commo data
        try:
            self.get_como()
        except:
            pass
        # Create kala data
        try:
            self.get_kala()
        except:
            pass

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

    def plot_pe_opt(self, opt, y_s=1400, m_s=1, y_e=1401, m_e=12):
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
        history = pd.read_excel(f"{MACROPATH}/my.xlsx")
        # call year_1400_data:
        data_1400 = pd.read_excel(f"{MACROPATH}/my.xlsx", sheet_name="1400")
        data_1401 = pd.read_excel(f"{MACROPATH}/my.xlsx", sheet_name="1401")
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
            "us_inflation",
        ]
        yearly_col = [
            "month",
            "base_money",
            "paper_money",
            "current_deposits",
            "non_current_deposits",
            "cash",
            "total_export",
            "import",
            "cpi",
            "ppi",
        ]
        # rename coloumns
        for i in range(len(history.columns)):
            history.rename(columns={history.columns[i]: my_col[i]}, inplace=True)
        # rename 1400_columns
        for i in range(len(data_1400.columns)):
            data_1400.rename(
                columns={data_1400.columns[i]: yearly_col[i]}, inplace=True
            )
            data_1401.rename(
                columns={data_1401.columns[i]: yearly_col[i]}, inplace=True
            )
        # set index
        history.set_index("year", inplace=True)
        data_1400.set_index("month", inplace=True)
        data_1401.set_index("month", inplace=True)
        # create data_1400_dollar
        ir_1400 = []
        pe_1400 = []
        dollar_azad_1400 = []
        dollar_nima_1400 = []
        for i in data_1400.index:
            if i < 12:
                date1 = f"1400-{i:02}"
                date2 = f"1400-{i+1:02}"
            else:
                date1 = f"1400-{i:02}"
                date2 = "1401-01"
            dollar_azad_1400.append(self.dollar_azad.loc[date1:date2]["Close"].mean())
            dollar_nima_1400.append(self.dollar_nima.loc[date1:date2]["Close"].mean())
            ir_1400.append(self.IR.loc[date1:date2]["Close"].mean())
            pe_1400.append(self.pe.loc[date1:date2]["Close"].mean())
        data_1400["dollar_azad"] = dollar_azad_1400
        data_1400["dollar_nima"] = dollar_nima_1400
        data_1400["pe"] = pe_1400
        data_1400["ir"] = ir_1400
        # create data_1401_dollar
        ir_1401 = []
        pe_1401 = []
        dollar_azad_1401 = []
        dollar_nima_1401 = []
        for i in data_1401.index:
            if i < 12:
                date1 = f"1401-{i:02}"
                date2 = f"1401-{i+1:02}"
            else:
                date1 = f"1401-{i:02}"
                date2 = "1402-01"
            dollar_azad_1401.append(self.dollar_azad.loc[date1:date2]["Close"].mean())
            dollar_nima_1401.append(self.dollar_nima.loc[date1:date2]["Close"].mean())
            ir_1401.append(self.IR.loc[date1:date2]["Close"].mean())
            pe_1401.append(self.pe.loc[date1:date2]["Close"].mean())
        data_1401["dollar_azad"] = dollar_azad_1401
        data_1401["dollar_nima"] = dollar_nima_1401
        data_1401["pe"] = pe_1401
        data_1401["ir"] = ir_1401
        data_1401["cash/base"] = data_1401["cash"] / data_1401["base_money"]
        data_1401["ratio_deposits"] = data_1401["current_deposits"] / data_1401["cash"]
        data_1400["cash/base"] = data_1400["cash"] / data_1400["base_money"]
        data_1400["ratio_deposits"] = data_1400["current_deposits"] / data_1400["cash"]
        data_1401.fillna(method="ffill", inplace=True)
        # add future data
        # create interest_ yearly data
        yearly_interest = []
        yearly_pe = []
        for i in history.index:
            date_1 = pd.to_datetime(JalaliDate(i, 1, 1).to_gregorian())
            date_2 = pd.to_datetime(JalaliDate(i, 12, 29).to_gregorian())
            yearly_interest.append(self.IR["Close"].loc[date_1:date_2].mean())
            yearly_pe.append(self.pe["Close"].loc[date_1:date_2].mean())
        history["IR"] = yearly_interest
        history["pe"] = yearly_pe
        history.loc[1401, "dollar"] = self.dollar_azad.iloc[-1]["Close"]
        history.loc[1401, "cpi"] = data_1401["cpi"].iloc[-1]
        history.loc[1401, "IR"] = self.IR["Close"].iloc[-1]
        history.loc[1401, "stock"] = self.shakhes_kol["Close"].iloc[-1]
        history.loc[1401, "pe"] = self.pe["Close"].iloc[-1]
        for i in [
            "cash",
            "current_deposits",
            "non_current_deposits",
            "paper_money",
            "base_money",
        ]:
            month = data_1401.index[-1]
            ratio = history.loc[1400, i] / data_1400[i].loc[month]
            history.loc[1401, i] = ratio * data_1401.loc[month, i]
        for i in ["import", "total_export"]:
            month = data_1401.index[-1]
            ratio = history.loc[1400, i] / data_1400[i].loc[0:month].sum()
            history.loc[1401, i] = ratio * data_1401[i].loc[0:month].sum()

        # extract monetary index
        monetary = history[
            [
                "base_money",
                "paper_money",
                "current_deposits",
                "non_current_deposits",
                "cash",
                "dollar",
            ]
        ]
        monetary["ratio_deposits"] = monetary["current_deposits"] / monetary["cash"]
        monetary["cash/base"] = monetary["cash"] / monetary["base_money"]
        # extract price index
        price = history[["dollar", "cpi", "ppi", "land", "dollar_land", "stock"]]
        price_90 = price.loc[1390:]
        # extract foreign exchange data
        exchange = history[
            [
                "dollar",
                "oil_export",
                "total_export",
                "import",
                "constant_gdp",
                "cpi",
                "cash",
                "IR",
                "pe",
                "us_inflation",
            ]
        ]

        # create return of data
        monetary_ret = pd.DataFrame(columns=monetary.columns)
        for i in monetary.columns:
            monetary_ret[i] = monetary[i].pct_change()
        data_1400_ret = pd.DataFrame(columns=data_1400.columns)
        for i in data_1400.columns:
            data_1400_ret[i] = data_1400[i].pct_change()
        data_1401_ret = pd.DataFrame(columns=data_1401.columns)
        for i in data_1401.columns:
            data_1401_ret[i] = data_1401[i].pct_change()
        # add value to exchange
        d = []
        lst_gdp = []
        lst_cash = []
        exchange.fillna(method="ffill", inplace=True)
        for i in range(len(exchange.index) - 1):
            cash_diff = (exchange.iloc[i + 1]["cash"] / exchange.iloc[i]["cash"]) - 1
            cpi_diff = (exchange.iloc[i + 1]["cpi"] / exchange.iloc[i]["cpi"]) - 1
            gdp_diff = (
                exchange.iloc[i + 1]["constant_gdp"] / exchange.iloc[i]["constant_gdp"]
            ) - 1
            lst_gdp.append(gdp_diff)
            lst_cash.append(cash_diff)
            factor = (
                (max(cash_diff, cpi_diff) - gdp_diff)
                + 1
                - (exchange.iloc[i + 1]["us_inflation"])
            )
            d.append(factor * exchange.iloc[i]["dollar"])
        d.insert(0, 8019)
        exchange["pred"] = d
        exchange_ret = pd.DataFrame(columns=["dollar", "cpi"])
        exchange_ret["dollar"] = exchange["dollar"].pct_change()
        exchange_ret["gdp"] = exchange["constant_gdp"].pct_change()
        exchange_ret["cpi"] = exchange["cpi"].pct_change()
        exchange_ret["cash"] = exchange["cash"].pct_change()
        exchange_ret["oil_export"] = exchange["oil_export"].pct_change()
        exchange_ret["total_export"] = exchange["total_export"].pct_change()
        exchange_ret["IR"] = exchange["IR"].pct_change()
        exchange_ret["pe"] = exchange["pe"].pct_change()
        price_ret = pd.DataFrame(columns=["ppi", "cpi"])
        price_ret["cpi"] = price["cpi"].pct_change()
        price_ret["ppi"] = price["ppi"].pct_change()
        price_ret["land"] = price["land"].pct_change()
        price_ret["dollar"] = price["dollar"].pct_change()
        price_ret["stock"] = price["stock"].pct_change()
        # create_uni var
        price_uni = pd.DataFrame(columns=price.columns)
        for i in price_uni.columns:
            price_uni[i] = price[i] / price[i].iloc[0]
        price_90_uni = pd.DataFrame(columns=price.columns)
        for i in price_90_uni.columns:
            price_90_uni[i] = price_90[i] / price_90[i].iloc[0]
        # dropna from data
        price_ret.dropna(inplace=True)
        # exchange_ret.dropna(inplace=True)
        monetary_ret.dropna(inplace=True)
        # send data to self
        self.history = history
        self.monetary = monetary
        self.price = price
        self.exchange = exchange
        self.exchange_ret = exchange_ret
        self.price_ret = price_ret
        self.monetary_ret = monetary_ret
        self.data_1400 = data_1400
        self.data_1400_ret = data_1400_ret
        self.data_1401 = data_1401
        self.data_1401_ret = data_1401_ret
        self.price_uni = price_uni
        self.price_90 = price_90
        self.price_90_uni = price_90_uni

    def get_como(self):
        como = pd.read_excel(
            f"{MACROPATH}/commoditie/monthly.xlsx", sheet_name="Monthly Prices"
        )
        for i in como.index:
            for j in como.columns:
                if como.loc[i, j] == "Crude oil, average":
                    x = i
        my_col = como.loc[x].values
        my_col[0] = "date"
        for i in range(len(como.columns)):
            como.rename(columns={como.columns[i]: my_col[i]}, inplace=True)
        como.drop([x, x + 1], inplace=True)
        como = como[-250:]
        como.set_index(["date"], inplace=True)
        como = como[
            [
                "Urea ",
                "Gold",
                "Crude oil, average",
                "Crude oil, Dubai",
                "Crude oil, WTI",
                "Iron ore, cfr spot",
                "Natural gas, US",
                "Natural gas, Europe",
                "Coal, Australian",
                "Copper",
                "Aluminum",
                "Zinc",
                "Potassium chloride **",
            ]
        ]

        my_col = [
            "urea",
            "gold",
            "oil_average",
            "oil_dubai",
            "oil_wti",
            "iron_ore",
            "gas_us",
            "gas_europe",
            "coal",
            "cooper",
            "aluminum",
            "zink",
            "kcl",
        ]
        # change column name
        for i in range(len(my_col)):
            como.rename(columns={como.columns[i]: my_col[i]}, inplace=True)

        # create como_uni
        como_uni = pd.DataFrame(columns=como.columns)
        for i in como.columns:
            como_uni[i] = como[i] / como[i].iloc[0]
        # send data to self
        self.como = como
        self.como_uni = como_uni

    def get_kala(self):
        kala = pd.read_excel(f"{MACROPATH}/physical.xlsx")
        self.kala = kala


class Stock:
    def __init__(
        self,
        Name,
        start_date="1396-01-01",
        end_date="1403-01-01",
        start_tester="1396-01-01",
        end_tester="1403-01-01",
        start_train="1396-01-01",
        end_train="1403-01-01",
        discounted_n=0.7,
        rf=0.3,
        erp=0.15,
        material="default",
    ):
        self.discounted_n = discounted_n
        self.Name = Name
        self.ful_name = wl_prod[Name]["name"]
        self.industry = wl_prod[Name]["indus"]
        self.farsi = wl_prod[Name]["token"]
        self.start_date = start_date
        self.start_tester = start_tester
        self.end_tester = end_tester
        self.start_train = start_train
        self.end_train = end_train
        self.end_date = end_date
        self.material = material
        self.tc = 0.012
        error = []
        self.error = error
        fiscal_dic = {
            12: {1: 3, 2: 6, 3: 9, 4: 12},
            9: {1: 12, 2: 3, 3: 6, 4: 9},
            6: {1: 9, 2: 12, 3: 3, 4: 6},
            3: {1: 6, 2: 9, 3: 12, 4: 3},
            10: {1: 1, 2: 4, 3: 7, 4: 10},
            8: {1: 11, 2: 2, 3: 5, 4: 8},
        }
        self.fiscal_dic = fiscal_dic
        self.rf = rf
        self.erp = erp
        ######## load price data ##########
        try:
            self.Price, self.Price_dollar = read_stock(
                self.farsi, self.start_date, self.end_date
            )
        except Exception as err:
            error.append((f"cant find {self.Name} price:{err}"))
        ######## load income  ############

        try:
            self.get_income("yearly")
            self.get_income("quarterly")
        except:
            error.append(f"cant find {self.Name}  income")
        ############# Download Buy_Power ##############
        try:
            self.Buy_Power = type_record(self.farsi)
            self.Buy_Power_w = self.Buy_Power.resample("W").mean()
            self.Buy_Power_m = self.Buy_Power.resample("M").mean()
        except:
            error.append(f"cant find {self.Name}  buy power")
        try:
            self.dollar_analyse = (
                self.income_dollar_yearly[["فروش"]]
                / self.income_dollar_yearly.iloc[0]["فروش"]
            )
        except:
            error.append(f"cant find {self.Name}  dollar analyse")
        try:
            self.n = len(self.income_quarterly.index)
        except:
            error.append(f"cant find {self.Name}  income_quarterly")
        ######### Load balancesheet ############
        try:
            self.get_balance_sheet("yearly")
            self.get_balance_sheet("quarterly")
        except Exception as err:
            error.append(f"add balance sheet {self.Name} : {err}")
        ########## Load cash_flow ##############
        try:
            self.get_cash_flow("yearly")
            self.get_cash_flow("quarterly")
        except Exception as err:
            error.append(f"add cash_flow {self.Name} : {err}")

        try:
            self.dollar_analyse["سود (زیان) خالص"] = (
                self.income_dollar_yearly[["سود (زیان) خالص"]]
                / self.income_dollar_yearly.iloc[0]["سود (زیان) خالص"]
            )
            self.dollar_analyse["Total_change"] = self.income_dollar_yearly[
                ["فروش"]
            ].pct_change()
            self.dollar_analyse["Net_change"] = self.income_dollar_yearly[
                ["سود (زیان) خالص"]
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
        except Exception as err:
            error.append(f"cant create {self.Name}  dollar analyse {err}")
        ####### Create_volume_profile ##########
        try:
            self.Vp, self.Price_bin = voloume_profile(self.Price, "2020", 60)
        except Exception as err:
            error.append(f"cant find {self.Name} voloume profile {err}")
        ############ create tester module #############
        try:
            self.my_tester = TesterOneSide(
                self.Price, self.tc, self.Name, self.start_train, self.end_train
            )
        except Exception as err:
            error.append(f"cant create tester {err}")

        ############ Load P/E Historical #############
        try:
            self.pe, self.pe_n, self.pe_u = get_pe_data(self.Name)
            self.dollar_azad, self.dollar_nima = read_dollar(
                self.start_date, self.end_date
            )
        except Exception as err:
            error.append(f"cant create p/e historical {err}")
        ## ############### load product ##################
        try:
            self.get_product("yearly")
            self.get_product("monthly")
            self.get_product("quarterly")
            # self.pre_process_product_data()
        except Exception as err:
            error.append(f"add prouct data {self.Name} : {err}")

        ########### load cost ###########
        try:
            self.get_cost("yearly")
            self.get_cost("quarterly")
        except Exception as err:
            error.append(f"add cost {self.Name} : {err}")
        ######### Load group p/e data #########
        try:
            self.group, self.gropu_n, self.group_u = get_pe_data(self.industry, "index")
        except Exception as err:
            error.append(f"cant load p/e group data {err}")
        ########## Load  Optimize_Strategy file ############
        try:
            opt = pd.read_excel(
                f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['opt']}"
            )
            # simulate with opt file
            self.my_tester.test_strategy(
                opt["SMA_s"].iloc[0],
                opt["SMA_l"].iloc[0],
                opt["VMA_S"].iloc[0],
                opt["VMA_l"].iloc[0],
                self.start_tester,
                self.end_tester,
            )
            self.opt = opt

        except Exception as err:
            print(f"add opt file {self.Name} : {err}")
            if self.industry != "fixed_income":
                self.my_tester.optimize_strategy(
                    range(3, 15, 3),
                    range(18, 40, 4),
                    range(10, 20, 4),
                    range(21, 60, 6),
                )
                opt = pd.read_excel(
                    f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['opt']}"
                )
                # simulate with opt file
                self.my_tester.test_strategy(
                    opt["SMA_s"].iloc[0],
                    opt["SMA_l"].iloc[0],
                    opt["VMA_S"].iloc[0],
                    opt["VMA_l"].iloc[0],
                    self.start_tester,
                    self.end_tester,
                )
                self.opt = opt
        ######## Create macro data(dollar rate) #########
        try:
            self.create_macro()
        except Exception as err:
            error.append(f"cant create_macro {self.Name} : {err}")
        ########### Predict future ############
        try:
            self.create_interest_data()
        except Exception as err:
            error.append(f"cant create_interest_data {err}")
        try:
            self.predict_parameter()
        except Exception as err:
            error.append(f"cant predict_parameters {err}")
        # try:
        #     #self.predict_revenue()
        # except Exception as err:
        #     error.append(f"cant predict_revenue {err}")
        # try:
        #     #self.predict_income()
        # except Exception as err:
        #     error.append(f"cant predict_income {err}")
        # try:
        #     #self.create_fcfe()
        # except Exception as err:
        #     error.append(f" cant create_fcfe {err}")
        # ########### add risk of falling ############
        # try:
        #     self.risk_falling = len(self.pe[self.pe["P/E-ttm"] < self.pe_fw]) / len(
        #         self.pe["P/E-ttm"]
        # )
        except Exception as err:
            error.append(f"cant calculate risk of falling {self.Name} : {err}")

        ############# Create end_data ##############
        try:
            self.create_end_data()
        except Exception as err:
            error.append(f" cant create_end_data {err}")
        ############ Create eps data ##########
        try:
            self.create_eps_data()
        except Exception as err:
            error.append(f" cant create_eps_data {err}")
        ############ add valueation of stock ###########
        try:
            self.predict_value()
        except Exception as err:
            error.append(f"cant valuation of {self.Name} : {err}")
        try:
            self.create_end_data()
        except Exception as err:
            error.append(f" cant create_end_data {err}")

        ########## Create_value_stategy_tester ##########
        try:
            self.value_tester = ValueStrategy(
                self.pe_forward_adjust,
                self.tc,
                self.start_train,
                self.end_train,
                self.Name,
            )
        except Exception as err:
            error.append(f"cant create value_tester {err}")

        ######## Load and optimize Value strategy_file #######
        try:
            value_opt = pd.read_excel(
                f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['value_opt']}"
            )
            self.value_tester.test_strategy(
                value_opt["sma_s"].iloc[0],
                value_opt["sma_l"].iloc[0],
                self.start_tester,
                self.end_tester,
            )
        except:
            try:
                self.value_tester.optimize_strategy(range(3, 30, 3), range(31, 60, 3))
                value_opt = pd.read_excel(
                    f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['value_opt']}"
                )
                self.value_tester.test_strategy(
                    value_opt["sma_s"].iloc[0],
                    value_opt["sma_l"].iloc[0],
                    self.start_tester,
                    self.end_tester,
                )
                self.value_opt = value_opt
            except Exception as err:
                error.append(f"cant optimize value_tester {err}")

        self.error = error

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

    def plot_voloume_profile(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10])
        ax1.barh(self.Price_bin, self.Vp, 100, edgecolor="black")
        ax1.grid(color="white")
        ax2.plot(self.Price["2020":]["Close"], linewidth=3)
        ax2.grid(color="white")

    def plot_pe(self, date1="1396-01-01", date2="1403-01-01"):
        df = self.pe_forward_adjust.loc[date1:date2]
        plt.figure(figsize=[15, 12])
        plt.subplot(3, 1, 1)
        plt.plot(df["eps"])
        plt.title(f"EPS_{self.Name}")
        plt.subplot(3, 1, 2)
        plt.hist(df["pe_adjust"], edgecolor="black", bins=50)
        plt.axvline(df["pe_adjust"].iloc[-1], color="red")
        plt.axvline(df["pe_adjust"].median(), color="red", linestyle="dashed")
        plt.subplot(3, 1, 3)
        plt.plot(df["price"])
        plt.plot(df["value"])

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
            self.dollar_income["سود (زیان) خالص_cret"],
            marker="o",
            color="blue",
            label="سود (زیان) خالص",
        )
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(
            self.dollar_income["Market_cret"], marker="o", color="black", label="Market"
        )
        plt.plot(
            self.dollar_income["سود (زیان) خالص_cret"],
            marker="o",
            color="blue",
            label="سود (زیان) خالص",
        )
        plt.legend()
        plt.figure(figsize=[20, 8])
        plt.subplot(1, 3, 1)
        plt.scatter(self.dollar_income["dollar"], self.dollar_income["سود (زیان) خالص"])
        plt.title("dollar and سود (زیان) خالص")
        plt.subplot(1, 3, 2)
        plt.scatter(self.dollar_income["dollar"], self.dollar_income["Net_margin"])
        plt.title("dollar and Net_margin")
        plt.subplot(1, 3, 3)
        plt.scatter(self.dollar_income["dollar"], self.dollar_income["Market"])
        plt.title("dollar and Market")

    def predict_income(self):
        df_income = self.income_yearly.iloc[[-1]].copy()
        df_income.loc[self.future_year] = 0
        df_income.loc[self.future_year + 1] = 0
        pred_income = IncomeDataFrame(df_income, self.pred_cost)
        pred_income.loc[self.future_year, "فروش"] = self.pred_revenue.loc[
            self.future_year
        ].values[0]
        pred_income.loc[self.future_year + 1, "فروش"] = self.pred_revenue.loc[
            self.future_year + 1
        ].values[0]
        pred_income.loc[
            self.future_year, "هزینه های عمومی, اداری و تشکیلاتی"
        ] = -self.pred_opex.loc[self.future_year].values[0]
        pred_income.loc[
            self.future_year + 1, "هزینه های عمومی, اداری و تشکیلاتی"
        ] = -self.pred_opex.loc[self.future_year + 1].values[0]
        pred_income.loc[
            self.future_year, "خالص سایر درامدها (هزینه ها) ی عملیاتی"
        ] = self.pred_other_op.loc[self.future_year].values[0]
        pred_income.loc[
            self.future_year + 1, "خالص سایر درامدها (هزینه ها) ی عملیاتی"
        ] = self.pred_other_op.loc[self.future_year + 1].values[0]
        pred_income.loc[
            self.future_year, "خالص سایر درامدها و هزینه های غیرعملیاتی"
        ] = self.pred_other_non_op.loc[self.future_year].values[0]
        pred_income.loc[
            self.future_year + 1, "خالص سایر درامدها و هزینه های غیرعملیاتی"
        ] = self.pred_other_non_op.loc[self.future_year + 1].values[0]
        pred_income.loc[self.future_year, "سرمایه"] = self.income_quarterly[
            "سرمایه"
        ].iloc[-1]
        pred_income.loc[self.future_year + 1, "سرمایه"] = self.income_quarterly[
            "سرمایه"
        ].iloc[-1]
        pred_income.update_dependent_columns()
        self.pred_income = pred_income

    def plot_revenue(self):
        plt.figure()
        df = self.price_revenue_yearly.copy()
        df.drop(["total", "جمع"], axis=1, inplace=True)
        arabic_text = list(df.columns)
        reshaped_text = map(lambda x: arabic_reshaper.reshape(x), arabic_text)
        bidi_text = map(lambda x: get_display(x), reshaped_text)
        plt.pie(df.iloc[-1], labels=list(bidi_text), autopct="%2.2f")
        plt.title(f"{self.Name}")
        plt.figure(figsize=[20, 15])
        plt.subplot(3, 1, 1)
        if self.industry != "palayesh":
            plt.plot(self.rate_monthly[self.major_good].iloc[-20:], marker="o")
        else:
            plt.plot(self.rate_quarterly[self.major_good].iloc[-20:], marker="o")
        plt.axhline(
            self.pred_rate[self.major_good].loc[self.future_year], linestyle="dashed"
        )
        reshaped_text = arabic_reshaper.reshape(self.major_good)
        persian_text = get_display(reshaped_text)
        plt.title(f"rate {self.Name} _ {persian_text}")
        plt.subplot(3, 1, 2)
        if self.industry != "palayesh":
            plt.plot(self.count_revenue_monthly[self.major_good].iloc[-20:], marker="o")
        else:
            plt.plot(
                self.count_revenue_quarterly[self.major_good].iloc[-20:], marker="o"
            )
        plt.title(f"count revenue {self.Name} _ {persian_text}")
        plt.subplot(3, 1, 3)
        if self.industry != "palayesh":
            plt.plot(self.rate_dollar_nima_monthly[self.major_good].iloc[-12:])
        else:
            plt.plot(self.rate_dollar_nima_quarterly[self.major_good].iloc[-12:])
        plt.axhline(
            self.pred_dollar_rate[self.major_good].loc[self.future_year],
            linestyle="dashed",
        )

    def create_eps_data(self):
        adress = f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['eps']}"
        df = pd.read_excel(adress, engine="openpyxl", index_col="year")
        df = df[::-1]
        # normalize_eps ans dps data
        for i in df.index:
            for j in df.columns:
                if df.loc[i][j] == "-":
                    df.loc[i, j] = np.nan
        df.dropna(inplace=True)
        df["capital_now"] = self.income_quarterly.iloc[-1]["سرمایه"] * np.ones(
            [len(df["capital_now"]), 1]
        )
        df["EPS"] = df["EPS"] * df["capital"] / df["capital_now"]
        df["DPS"] = df["DPS"] * df["capital"] / df["capital_now"]
        df["ratio"] = df["DPS"] / df["EPS"]

        ratio_mean = df["ratio"].median()
        # add future_year
        future_year = df.index[-1] + 1
        df.loc[future_year] = np.zeros(len(df.iloc[0]))
        df.loc[future_year, "EPS"] = self.pred_income.loc[future_year][
            "سود هر سهم بر اساس آخرین سرمایه"
        ]
        df.loc[future_year, "DPS"] = (
            self.pred_income.loc[future_year]["سود هر سهم بر اساس آخرین سرمایه"]
            * ratio_mean
        )
        df.loc[future_year, "capital"] = self.income_quarterly.iloc[-1]["سرمایه"]
        df.loc[future_year, "capital_now"] = self.income_quarterly.iloc[-1]["سرمایه"]
        df.loc[future_year, "ratio"] = (
            df.loc[future_year]["DPS"] / df.loc[future_year]["EPS"]
        )

        # add future year + 1
        df.loc[future_year + 1] = np.zeros(len(df.iloc[0]))
        df.loc[future_year + 1, "EPS"] = self.pred_income.loc[future_year + 1][
            "سود هر سهم بر اساس آخرین سرمایه"
        ]
        df.loc[future_year + 1, "DPS"] = (
            self.pred_income.loc[future_year + 1]["سود هر سهم بر اساس آخرین سرمایه"]
            * ratio_mean
        )
        df.loc[future_year + 1, "capital"] = self.income_quarterly.iloc[-1]["سرمایه"]
        df.loc[future_year + 1, "capital_now"] = self.income_quarterly.iloc[-1][
            "سرمایه"
        ]
        df.loc[future_year + 1, "ratio"] = (
            df.loc[future_year + 1]["DPS"] / df.loc[future_year + 1]["EPS"]
        )

        # add ret and cret
        df["EPS_ret"] = df["EPS"].pct_change()
        df["EPS_cret"] = df["EPS"] / df["EPS"].iloc[0]
        self.eps_data = df
        self.grow_eps = self.eps_data["EPS_ret"].iloc[-1]

    def get_balance_sheet(self, periode):
        # create adress and coloumns proper tp peride
        adress = (
            f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['balance'][periode]}"
        )
        df = pd.read_excel(adress)
        if periode == "yearly":
            col = []
            all_time_id = re.findall(regex_en_timeid_q, str(df.loc[6]))
            for i in all_time_id:
                col.append(int(i[:4]))
            col.insert(0, "Data")
        elif periode == "quarterly":
            col = create_quaretrly_columns(df, self.fiscal_year)
        asset = select_df(df, "دوره مالی", "جمع داراییها")
        asset.dropna(inplace=True)
        asset.columns = col
        asset.set_index("Data", inplace=True)
        if "دوره مالی" in asset.index:
            asset.drop("دوره مالی", inplace=True)
        if "تاریخ انتشار" in asset.index:
            asset.drop("تاریخ انتشار", inplace=True)
        asset = asset.T
        debt = select_df(
            df, "پرداختنی‌های تجاری و سایر پرداختنی‌ها", "جمع بدهیهای جاری و غیر جاری"
        )
        debt.columns = col
        debt.set_index("Data", inplace=True)
        debt = debt.T
        equity = select_df(
            df, "بدهیها و حقوق صاحبان سهام", "جمع بدهیها و حقوق صاحبان سهام"
        )
        equity.columns = col
        equity.set_index("Data", inplace=True)
        equity.dropna(inplace=True)
        equity = equity.T
        if periode == "yearly":
            self.asset_yearly = asset
            self.debt_yearly = debt
            self.equity_yearly = equity
        if periode == "quarterly":
            self.asset_quarterly = asset
            self.debt_quarterly = debt
            self.equity_quarterly = equity

    def get_cash_flow(self, period):
        adress = f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['cash'][period]}"
        df = pd.read_excel(adress)
        if period == "yearly":
            col = []
            all_time_id = re.findall(regex_en_timeid_q, str(df.loc[6]))
            for i in all_time_id:
                col.append(int(i[:4]))
            col.insert(0, "Data")
        elif period == "quarterly":
            col = create_quaretrly_columns(df, self.fiscal_year)
        op_cash = select_df(df, "دوره مالی", "فعالیتهای سرمایه گذاری")
        op_cash.dropna(inplace=True)
        op_cash.columns = col
        op_cash.set_index("Data", inplace=True)
        op_cash.drop(["دوره مالی", "تاریخ انتشار"], inplace=True)
        op_cash = op_cash.T
        invest_cash = select_df(df, "فعالیتهای سرمایه گذاری", "فعالیتهای تامین مالی")
        invest_cash.dropna(inplace=True)
        invest_cash.columns = col
        invest_cash.set_index("Data", inplace=True)
        invest_cash = invest_cash.T

        finance_cash = select_df(df, "فعالیتهای تامین مالی", "مبادلات غیر نقدی")
        finance_cash.dropna(inplace=True)
        finance_cash.columns = col
        finance_cash.set_index("Data", inplace=True)

        finance_cash = finance_cash.T

        if period == "yearly":
            self.op_cash_yearly = op_cash
            self.invest_cash_yearly = invest_cash
            self.finance_cash_yearly = finance_cash
        if period == "quarterly":
            self.op_cash_quarterly = op_cash
            self.invest_cash_quarterly = invest_cash
            self.finance_cash_quarterly = finance_cash

    def predict_balance_sheet(self):
        # call data from self
        balance = self.balance_sheet_yearly.copy()
        cost = np.abs(self.income_yearly["Cost_of_Revenue"])
        tangible = self.tangible.copy()
        future_year = balance.index[-1] + 1
        receivable_rev_ratio = (
            self.balance_sheet_yearly["short term receivables"]
            / self.income_yearly["فروش"]
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
            receivable_rev_ratio_mean * self.pred_income.loc[future_year]["فروش"]
        )
        balance.loc[future_year + 1]["short term receivables"] = (
            receivable_rev_ratio_mean * self.pred_income.loc[future_year + 1]["فروش"]
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
            self.balance_sheet_yearly["pre received"] / self.income_yearly["فروش"]
        )
        prerecieved_rev_ratio_mean = np.average(prerecieved_rev_ratio[-2:], weights=w)
        # add prerecieved to data frame
        balance.loc[future_year]["pre received"] = (
            prerecieved_rev_ratio_mean * self.pred_income.loc[future_year]["فروش"]
        )
        balance.loc[future_year + 1]["pre received"] = (
            prerecieved_rev_ratio_mean * self.pred_income.loc[future_year + 1]["فروش"]
        )
        # add future year to tangible
        tangible.loc[future_year] = np.zeros(len(tangible.iloc[0]))
        tangible.loc[future_year, "first"] = tangible.loc[future_year - 1]["end"]
        tangible.loc[future_year, "add"] = np.median(
            tangible["add_cost_ratio"][1:-1]
        ) * np.abs(self.pred_income.loc[future_year]["Cost_of_Revenue"])
        tangible.loc[future_year, "depreciation"] = np.median(
            tangible["depreciation_ratio"][1:-1]
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
        tangible.loc[future_year + 1, "add"] = np.median(
            tangible["add_cost_ratio"][1:-1]
        ) * np.abs(self.pred_income.loc[future_year + 1]["Cost_of_Revenue"])
        tangible.loc[future_year + 1, "depreciation"] = np.median(
            tangible["depreciation_ratio"][1:-1]
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
            balance_com[i] = balance[i] / self.pred_income["فروش"]
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
        cost_dl = pd.read_excel(
            f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['cost'][period]}"
        )
        official_dl = pd.read_excel(
            f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['official'][period]}"
        )
        self.cost_dl = cost_dl
        if period == "yearly":
            # all data is cost_dl

            # select desired data
            cost = select_df(cost_dl, "بهای تمام شده", "جمع بهای تمام شده")
            overhead = select_df(cost_dl, "هزینه سربار", "جمع")
            official = select_df(official_dl, "هزینه های عمومی و اداری", "جمع")
            count_consump = select_df(cost_dl, "مقدار مصرف طی دوره", "جمع")
            price_consump = select_df(cost_dl, "مبلغ مصرف طی دوره", "جمع")
            count_buy = select_df(cost_dl, "مقدار خرید طی دوره", "جمع")
            price_buy = select_df(cost_dl, "مبلغ خرید طی دوره", "جمع")
            # define column
            all_time_id_cost = re.findall(regex_en_timeid_q, str(cost.loc[6]))
            cost_col = [int(i[:4]) for i in all_time_id_cost]
            cost_col.insert(0, "Data")
            all_time_id_official = re.findall(regex_en_timeid_q, str(official.loc[6]))
            official_col = [int(i[:4]) for i in all_time_id_official]
            official_col.insert(0, "Data")

        elif period == "quarterly":
            # select desired data
            cost = select_df(cost_dl, "بهای تمام شده", "جمع بهای تمام شده")
            overhead = select_df(cost_dl, "هزینه سربار", "جمع")
            official = select_df(official_dl, "هزینه های عمومی و اداری", "جمع")
            count_consump = select_df(cost_dl, "مقدار مصرف طی دوره", "جمع")
            price_consump = select_df(cost_dl, "مبلغ مصرف طی دوره", "جمع")
            count_buy = select_df(cost_dl, "مقدار خرید طی دوره", "جمع")
            price_buy = select_df(cost_dl, "مبلغ خرید طی دوره", "جمع")
            cost_col = create_quaretrly_columns(cost, self.fiscal_year)
            official_col = create_quaretrly_columns(official, self.fiscal_year)
        # preprocess data
        count_consump.drop("Unnamed: 2", axis=1, inplace=True)
        price_consump.drop("Unnamed: 2", axis=1, inplace=True)
        count_buy.drop("Unnamed: 2", axis=1, inplace=True)
        price_buy.drop("Unnamed: 2", axis=1, inplace=True)
        cost.dropna(how="all", inplace=True)
        official.dropna(how="all", inplace=True)
        overhead.dropna(how="all", inplace=True)
        count_consump.dropna(how="all", inplace=True)
        price_consump.dropna(how="all", inplace=True)
        count_buy.dropna(how="all", inplace=True)
        price_buy.dropna(how="all", inplace=True)
        # change column name
        cost.columns = cost_col
        overhead.columns = cost_col
        count_consump.columns = cost_col
        price_consump.columns = cost_col
        count_buy.columns = cost_col
        price_buy.columns = cost_col
        official.columns = official_col
        cost.dropna(axis=0, inplace=True)
        official.dropna(axis=0, inplace=True)
        overhead.dropna(axis=0, inplace=True)
        count_consump.dropna(axis=0, inplace=True)
        price_consump.dropna(axis=0, inplace=True)
        count_buy.dropna(axis=0, inplace=True)
        price_buy.dropna(axis=0, inplace=True)
        # set Data is index
        cost.set_index("Data", inplace=True)
        official.set_index("Data", inplace=True)
        overhead.set_index("Data", inplace=True)
        count_consump.set_index("Data", inplace=True)
        price_consump.set_index("Data", inplace=True)
        count_buy.set_index("Data", inplace=True)
        price_buy.set_index("Data", inplace=True)
        # drop unnessecary data
        cost.drop("بهای تمام شده", inplace=True)
        overhead.drop("هزینه سربار", inplace=True)
        official.drop("هزینه های عمومی و اداری", inplace=True)
        count_consump.drop("مقدار مصرف طی دوره", inplace=True)
        price_consump.drop("مبلغ مصرف طی دوره", inplace=True)
        count_buy.drop("مقدار خرید طی دوره", inplace=True)
        price_buy.drop("مبلغ خرید طی دوره", inplace=True)
        # transpose data
        cost = cost.T
        overhead = overhead.T
        official = official.T
        count_consump = count_consump.T
        price_consump = price_consump.T
        count_buy = count_buy.T
        price_buy = price_buy.T

        ### delet_empt#####
        delete_empty(count_consump)
        delete_empty(price_consump)
        delete_empty(count_buy)
        delete_empty(price_buy)
        delete_empty(cost)
        delete_empty(official)
        delete_empty(overhead)
        # merge_Same_columns
        merge_same_columns(count_consump)
        merge_same_columns(price_consump)
        merge_same_columns(count_buy)
        merge_same_columns(price_buy)
        # remove_zero_from_dataF
        count_consump = remove_zero(count_consump)
        price_consump = remove_zero(price_consump)
        count_buy = remove_zero(count_buy)
        price_buy = remove_zero(price_buy)
        cost = remove_zero(cost)
        official = remove_zero(official)
        overhead = remove_zero(overhead)
        # delete non same columns
        drop_non_same_columns(price_consump, count_consump)
        replace_negative_data(count_consump)
        replace_negative_data(price_consump)
        replace_negative_data(count_buy)
        replace_negative_data(price_buy)
        # merge_similar_columns
        count_consump = merge_similar_columns(count_consump)
        price_consump = merge_similar_columns(price_consump)
        count_buy = merge_similar_columns(count_buy)
        price_buy = merge_similar_columns(price_buy)
        drop_non_same_columns(price_consump, count_consump)
        drop_non_same_columns(price_buy, count_buy)

        drop_non_same_index(price_consump, count_consump)
        drop_non_same_index(price_buy, count_buy)
        ##### fill_outlier_data ########
        fill_outlier_data(count_consump)
        fill_outlier_data(price_consump)
        fill_outlier_data(count_buy)
        fill_outlier_data(price_buy)
        rate_consump = price_consump / count_consump
        rate_buy = price_buy / count_buy
        ##### delete negative data from cost #####
        cost[cost["دستمزد مستقیم تولید"] < 0] = cost["دستمزد مستقیم تولید"].median()
        cost[cost["مواد مستقیم مصرفی"] < 0] = cost["مواد مستقیم مصرفی"].median()
        ##### work with inf data ######
        rate_consump.replace([np.inf, -np.inf], np.nan, inplace=True)
        rate_consump.fillna(method="ffill", inplace=True)
        rate_consump.fillna(method="bfill", inplace=True)
        rate_buy.replace([np.inf, -np.inf], np.nan, inplace=True)
        rate_buy.fillna(method="ffill", inplace=True)
        rate_buy.fillna(method="bfill", inplace=True)
        for i in rate_consump.index:
            for j in rate_consump.columns:
                if (count_consump.loc[i, j] == 0.01) or (count_consump.loc[i, j] <= 0):
                    rate_consump.loc[i, j] = 0
        for i in rate_buy.index:
            for j in rate_buy.columns:
                if (count_buy.loc[i, j] == 0.01) or (count_buy.loc[i, j] <= 0):
                    rate_buy.loc[i, j] = 0

        # define new definition of cost extract units of cost
        converte_numeric(cost)
        converte_numeric(overhead)
        converte_numeric(official)
        converte_numeric(count_consump)
        converte_numeric(price_consump)
        converte_numeric(rate_consump)
        converte_numeric(count_buy)
        converte_numeric(price_buy)
        converte_numeric(rate_buy)
        # inventory ratio
        alpha = cost["جمع بهای تمام شده"] / cost["جمع"]
        # send data to self
        if period == "yearly":
            self.cost_yearly = cost
            self.overhead_yearly = overhead
            self.official_yearly = official
            self.count_consump_yearly = count_consump
            self.price_consump_yearly = price_consump
            self.rate_consump_yearly = rate_consump
            self.count_buy_yearly = count_buy
            self.price_buy_yearly = price_buy
            self.rate_buy_yearly = rate_buy
            self.inventory_ratio_yearly = alpha
        elif period == "quarterly":
            self.cost_quarterly = cost
            self.overhead_quarterly = overhead
            self.official_quarterly = official
            self.count_consump_quarterly = count_consump
            self.price_consump_quarterly = price_consump
            self.rate_consump_quarterly = rate_consump
            self.count_buy_quarterly = count_buy
            self.price_buy_quarterly = price_buy
            self.rate_buy_quarterly = rate_buy
            self.inventory_ratio_quarterly = alpha

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
            self.income_yearly[1:]["Cost_of_Revenue"].values
        )
        tangible["depreciation_ratio"] = tangible["depreciation"] / (
            tangible["add"] + tangible["first"]
        )
        self.tangible = tangible
        interest = pd.DataFrame(
            columns=["first", "add", "pay", "interest", "end"],
            index=self.income_yearly.index[1:],
        )

        interest["first"] = self.balance_sheet_yearly["financial facilities"][
            :-1
        ].values
        interest["end"] = self.balance_sheet_yearly["financial facilities"][1:].values
        interest["interest"] = np.abs(self.income_yearly["Interest_Expense"][1:].values)
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
        # remove negative add inv ratio
        for i in interest.index:
            if interest.loc[i, "add_inv_ratio"] < 0:
                interest.loc[i, "add_inv_ratio"] = 0
            # if interest.loc[i, "add_inv_ratio"] > 1:
            # interest.loc[i, "add_inv_ratio"] = 1
        interest["pay_ratio"] = interest["pay"] / (interest["first"] + interest["add"])
        interest["interest_ratio"] = interest["interest"] / (
            interest["first"] + interest["add"]
        )
        self.interest = interest
        self.tangible = tangible

    def predict_interest(self):
        w = [1, 3]
        future_year = self.income_yearly.index[-1] + 1
        self.future_year = future_year
        interest = self.interest
        # add future year to interest
        interest.loc[future_year] = np.zeros(len(interest.iloc[0]))
        interest.loc[future_year, "first"] = interest.loc[future_year - 1]["end"]
        # investing=capital expenditure+working capital
        inv = (
            self.pred_inv_balance.loc[future_year]["wc"]
            + self.tangible.loc[future_year]["add"]
        )
        if inv < 0:
            inv = 0
        interest.loc[future_year, "add_inv_ratio"] = np.median(
            interest["add_inv_ratio"][-4:-1]
        )
        interest.loc[future_year, "pay_ratio"] = np.median(interest["pay_ratio"][-4:-1])
        interest.loc[future_year, "add"] = (
            np.average(interest["add_inv_ratio"][-3:-1]) * inv
        )
        interest.loc[future_year, "pay"] = np.median(interest["pay_ratio"][-4:-1]) * (
            interest.loc[future_year, "first"] + interest.loc[future_year, "add"]
        )
        interest.loc[future_year, "interest"] = np.median(
            interest["interest_ratio"][-4:-1]
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
        interest.loc[future_year + 1, "add_inv_ratio"] = np.median(
            interest["add_inv_ratio"][-4:-1]
        )
        interest.loc[future_year + 1, "pay_ratio"] = np.median(
            interest["pay_ratio"][-4:-1]
        )
        interest.loc[future_year + 1, "add"] = (
            np.median(interest["add_inv_ratio"][-4:-1]) * inv
        )
        interest.loc[future_year + 1, "pay"] = np.median(
            interest["pay_ratio"][-4:-1]
        ) * (
            interest.loc[future_year + 1, "first"]
            + interest.loc[future_year + 1, "add"]
        )
        interest.loc[future_year + 1, "interest"] = np.median(
            interest["interest_ratio"][-4:-1]
        ) * (
            interest.loc[future_year + 1, "first"]
            + interest.loc[future_year + 1, "add"]
        )
        interest.loc[future_year + 1, "end"] = (
            interest.loc[future_year + 1, "first"]
            + interest.loc[future_year + 1, "add"]
            - interest.loc[future_year + 1, "pay"]
        )
        self.interest = interest
        # add interest to pred_income
        self.pred_income.loc[future_year, "Interest_Expense"] = -interest.loc[
            future_year, "interest"
        ]
        self.pred_income.loc[future_year + 1, "Interest_Expense"] = -interest.loc[
            future_year + 1, "interest"
        ]

    def create_fcfe(self):
        fcfe = pd.DataFrame(
            columns=["سود (زیان) خالص", "fcfe", "ratio"],
            index=self.pred_inv_balance.index,
        )
        fcfe["سود (زیان) خالص"] = self.pred_income["سود (زیان) خالص"][1:].values
        fcfe["fcfe"] = (
            fcfe["سود (زیان) خالص"]
            + self.tangible["depreciation"]
            - self.tangible["add"]
            - self.pred_inv_balance["wc"]
        )
        fcfe["ratio"] = fcfe["fcfe"] / fcfe["سود (زیان) خالص"]
        self.fcfe = fcfe
        self.p_fcfe = self.pe_fw / self.fcfe.loc[self.future_year]["ratio"]

    def get_product(self, period):
        product_dl = pd.read_excel(
            f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['product'][period]['']}"
        )
        if period == "yearly":
            # select desired data
            try:
                count_product = select_df(product_dl, "مقدار تولید", "جمع")
            except Exception as err:
                self.error.append(f"cant create count_product yearly {err}")
            count_revenue = select_df(product_dl, "مقدار فروش", "جمع")
            price_revenue = select_df(product_dl, "مبلغ فروش", "جمع")
            # define column
            all_time_id = re.findall(regex_en_timeid_q, str(product_dl.loc[6]))
            my_col = []
            for i in all_time_id:
                my_col.append(int(i[:4]))
            my_col.insert(0, "Data")
            my_col.insert(1, "unit")

        if period == "monthly":
            # selece desired data
            try:
                count_product = select_df(product_dl, "مقدار تولید", "جمع")
            except Exception as err:
                self.error.append(f"cant create count_product monthly {err}")
            count_revenue = select_df(product_dl, "مقدار فروش", "جمع")
            price_revenue = select_df(product_dl, "مبلغ فروش", "جمع")
            # define my_col
            all_time_id = re.findall(regex_en_timeid_q, str(count_product.iloc[0]))
            my_col = all_time_id
            my_col.insert(0, "Data")
            my_col.insert(1, "unit")
        if period == "quarterly":
            # select desired data
            try:
                count_product = select_df(product_dl, "مقدار تولید", "جمع")
            except Exception as err:
                self.error.append(f"cant create count_product quarterly {err}")
            count_revenue = select_df(product_dl, "مقدار فروش", "جمع")
            price_revenue = select_df(product_dl, "مبلغ فروش", "جمع")
            # define column
            my_col = create_quaretrly_columns(product_dl, self.fiscal_year)
            my_col.insert(1, "unit")
        # change column name
        for i in range(len(my_col)):
            count_product.rename(
                columns={count_product.columns[i]: my_col[i]}, inplace=True
            )
            count_revenue.rename(
                columns={count_revenue.columns[i]: my_col[i]}, inplace=True
            )
            price_revenue.rename(
                columns={price_revenue.columns[i]: my_col[i]}, inplace=True
            )
        # set Data is index
        count_product.set_index("Data", inplace=True)
        count_revenue.set_index("Data", inplace=True)
        price_revenue.set_index("Data", inplace=True)
        # delete unnecessary data'
        count_product.dropna(how="all", inplace=True)
        count_product.drop("مقدار تولید", inplace=True)
        count_revenue.dropna(how="all", inplace=True)
        count_revenue.drop("مقدار فروش", inplace=True)
        price_revenue.dropna(how="all", inplace=True)
        price_revenue.drop("مبلغ فروش", inplace=True)
        # transpose data
        count_product = count_product.T
        count_revenue = count_revenue.T
        price_revenue = price_revenue.T
        # extract unit and delete from df
        self.unit_prod = count_product.loc[["unit"]]
        count_product.drop("unit", inplace=True)
        count_revenue.drop("unit", inplace=True)
        price_revenue.drop("unit", inplace=True)
        # delete empty file
        delete_empty(count_product)
        delete_empty(count_revenue)
        delete_empty(price_revenue)
        # merge_same_index
        merge_same_index(count_product)
        merge_same_index(count_revenue)
        merge_same_index(price_revenue)
        # merge_same_columns
        merge_same_columns(count_product)
        merge_same_columns(count_revenue)
        merge_same_columns(price_revenue)
        ### Convert_numeric #####
        converte_numeric(count_product)
        converte_numeric(count_revenue)
        converte_numeric(price_revenue)
        # remove_zero from data
        count_product = remove_zero(count_product)
        count_revenue = remove_zero(count_revenue)
        price_revenue = remove_zero(price_revenue)

        # same columns
        drop_non_same_columns(price_revenue, count_revenue)
        price_revenue["total"] = price_revenue.sum(axis=1) - price_revenue["جمع"]
        count_revenue["total"] = count_revenue.sum(axis=1) - count_revenue["جمع"]

        # Merge similar columns
        count_product = merge_similar_columns(count_product)
        count_revenue = merge_similar_columns(count_revenue)
        price_revenue = merge_similar_columns(price_revenue)

        # delete noise of count_revenue and price_revenue
        for i in count_revenue.index:
            for j in count_revenue.columns:
                if (count_revenue.loc[i, j] < 1) & (
                    (price_revenue.loc[i, j] != 0) | (price_revenue.loc[i, j] != 0.01)
                ):
                    price_revenue.loc[i, j] = 0.01

        # replace negative data in count_revenue
        replace_negative_data(count_revenue)
        ### fill outlier data ######
        fill_outlier_data(count_product)
        # fill_outlier_data(count_revenue)
        # fill_outlier_data(price_revenue)
        # remove_zero from data
        count_product = remove_zero(count_product)
        count_revenue = remove_zero(count_revenue)
        price_revenue = remove_zero(price_revenue)
        # create count_product_com
        count_product_com = pd.DataFrame(columns=count_product.columns)
        try:
            for i in count_product.columns:
                count_product_com[i] = count_product[i] / count_product["جمع"]
        except Exception as err:
            print(f"cant create count_product_com {self.Name} {period} : {err}")
        # create count_revenue_com
        try:
            count_revenue_com = pd.DataFrame(columns=count_revenue.columns)
            for i in count_revenue.columns:
                count_revenue_com[i] = count_revenue[i] / count_revenue["جمع"]
        except Exception as err:
            print(f"cant create count_revenue_com {self.Name} {period} : {err}")
        # create price_revenue_com
        try:
            price_revenue_com = pd.DataFrame(columns=price_revenue.columns)
            for i in price_revenue.columns:
                price_revenue_com[i] = price_revenue[i] / price_revenue["جمع"]
        except Exception as err:
            print(f"cant create price_revenuee_com {self.Name} {period} : {err}")

        # create price_revenue major
        try:
            price_revenue_com_major = pd.DataFrame(index=price_revenue_com.index)
            for i in price_revenue_com.index:
                for j in price_revenue_com.columns:
                    if price_revenue_com.loc[i, j] > 0.005:
                        price_revenue_com_major.loc[i, j] = price_revenue_com.loc[i, j]
            price_revenue_com_major.drop("جمع", axis=1, inplace=True)
            price_revenue_com_major["جمع"] = price_revenue_com_major.sum(axis=1)
            price_revenue_com_major.fillna(0, inplace=True)
            price_revenue_major = price_revenue[price_revenue_com_major.columns]
            count_revenue_major = count_revenue[price_revenue_com_major.columns]
        except Exception as err:
            price_revenue_com_major = 0
            count_revenue_major = 0
            price_revenue_major = 0
            print(f"cant create major data {self.Name} {period} : {err}")
        # create_rate
        try:
            rate = price_revenue / count_revenue
            ### work with inf rate #####
            rate.replace([np.inf, -np.inf], np.nan, inplace=True)
            rate.fillna(method="ffill", inplace=True)
            rate_major = price_revenue_major / count_revenue_major
            rate_change = rate.pct_change()
            rate_change.dropna(inplace=True)
        except Exception as err:
            print(f"cant create rate {self.Name} {period} : {err}")
            rate = 0
            rate_major = 0
        # fillna
        count_product.fillna(0, inplace=True)
        count_revenue.fillna(0, inplace=True)
        price_revenue.fillna(0, inplace=True)
        ############ send data to self ############
        if period == "yearly":
            self.count_product_yearly = count_product
            self.count_product_com_yearly = count_product_com
            # self.count_product_major_yearly=count_product_major

            self.count_revenue_yearly = count_revenue
            self.count_revenue_com_yearly = count_revenue_com
            self.count_revenue_major_yearly = count_revenue_major

            self.price_revenue_yearly = price_revenue
            self.price_revenue_com_yearly = price_revenue_com
            self.price_revenue_major_yearly = price_revenue_major
            self.price_revenue_com_major_yearly = price_revenue_com_major

            self.rate_yearly = rate
            self.rate_major_yearly = rate_major
            self.rate_change_yearly = rate_change
        if period == "monthly":
            self.count_product_monthly = count_product
            self.count_product_com_monthly = count_product_com
            # self.count_product_major_monthly=count_product_major

            self.count_revenue_monthly = count_revenue
            self.count_revenue_com_monthly = count_revenue_com
            self.count_revenue_major_monthly = count_revenue_major

            self.price_revenue_monthly = price_revenue
            self.price_revenue_com_monthly = price_revenue_com
            self.price_revenue_major_monthly = price_revenue_major
            self.price_revenue_com_major_monthly = price_revenue_com_major
            #### reindex monthly data ##########
            self.reindex_monthly_data("count")
            self.reindex_monthly_data("price")
            self.rate_monthly = self.price_revenue_monthly / self.count_revenue_monthly
            self.rate_monthly.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.rate_monthly.fillna(method="ffill", inplace=True)
            self.rate_monthly.fillna(method="bfill", inplace=True)
            self.rate_major_monthly = rate
            self.rate_change_monthly = rate_change
        if period == "quarterly":
            self.count_product_quarterly = count_product
            self.count_product_com_quarterly = count_product_com
            # self.count_product_major_quarterly=count_product_major

            self.count_revenue_quarterly = count_revenue
            self.count_revenue_com_quarterly = count_revenue_com
            self.count_revenue_major_quarterly = count_revenue_major

            self.price_revenue_quarterly = price_revenue
            self.price_revenue_com_quarterly = price_revenue_com
            self.price_revenue_major_quarterly = price_revenue_major
            self.price_revenue_com_major_quarterly = price_revenue_com_major

            self.rate_quarterly = rate
            self.rate_major_quarterly = rate_major
            self.rate_change_quarterly = rate_change

    def get_income(self, period):
        if period == "quarterly":
            adress = f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['income']['quarterly']['']['rial']}"
            df = pd.read_excel(adress)
            df = select_df(df, "دوره مالی", "سود هر سهم بر اساس آخرین سرمایه")
            col = create_quaretrly_columns(df, self.fiscal_year)
        if period == "yearly":
            adress = f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['income']['yearly']['']['rial']}"
            df = pd.read_excel(adress)
            df = select_df(df, "دوره مالی", "سود هر سهم بر اساس آخرین سرمایه")
            all_time_id = re.findall(regex_en_timeid_q, str(df.loc[6]))
            fiscal_year = int(all_time_id[-1][5:])
            self.fiscal_year = fiscal_year
            col = []
            for i in all_time_id:
                col.append(int(i[:4]))
            col.insert(0, "Data")
        df.columns = col
        df.set_index("Data", inplace=True)
        df.drop(["دوره مالی", "تاریخ انتشار"], inplace=True)
        df = remove_zero(df)
        df = df.T
        converte_numeric(df)
        df_com = df.div(df["فروش"], axis=0)
        if period == "yearly":
            self.income_yearly = df
            self.income_common_yearly = df_com
        if period == "quarterly":
            self.income_quarterly = df
            self.income_common_quarterly = df_com

        return df

    def create_macro(self):
        # macro economic data
        macro = Macro()
        #### create dollar yearly #########
        dollar_yearly = pd.DataFrame(
            index=self.income_yearly.index, columns=["azad", "nima"]
        )
        for i in dollar_yearly.index:
            if self.fiscal_year == 12:
                date_1 = f"{i}-01-01"
                date_2 = f"{i}-12-29"
            else:
                date_1 = f"{i-1}-{self.fiscal_year:02}-01"
                date_2 = f"{i}-{self.fiscal_year:02}-29"
            dollar_yearly.loc[i, "azad"] = macro.dollar_azad.loc[date_1:date_2][
                "Close"
            ].mean()
            dollar_yearly.loc[i, "nima"] = macro.dollar_nima.loc[date_1:date_2][
                "Close"
            ].mean()
        dollar_yearly["ratio"] = dollar_yearly["nima"] / dollar_yearly["azad"]

        self.dollar_yearly = dollar_yearly
        converte_numeric(self.dollar_yearly)
        #### create dollar quarterly #########
        dollar_quarterly = pd.DataFrame(
            index=self.income_quarterly.index, columns=["azad", "nima"]
        )
        for i in dollar_quarterly.index:
            year = int(i[:4])
            quarter = int(i[-1])
            month = self.fiscal_dic[self.fiscal_year][quarter]
            if month > self.fiscal_year:
                year = year - 1
            m1 = month - 2
            m2 = month
            if m1 != 0:
                date_1 = f"{year}-{m1:02}-01"
                date_2 = f"{year}-{m2:02}-29"
            else:
                date_1 = f"{year-1}-12-01"
                date_1 = f"{year}-{m2:02}-29"
            dollar_quarterly.loc[i, "azad"] = macro.dollar_azad.loc[date_1:date_2][
                "Close"
            ].mean()
            dollar_quarterly.loc[i, "nima"] = macro.dollar_nima.loc[date_1:date_2][
                "Close"
            ].mean()
        dollar_quarterly["ratio"] = dollar_quarterly["nima"] / dollar_quarterly["azad"]
        # converte_numeric(dollar_quarterly)
        self.dollar_quarterly = dollar_quarterly
        ###### Create_dollar_monthly #########
        dollar_monthly = pd.DataFrame(
            index=self.count_revenue_monthly.index, columns=["azad", "nima"]
        )
        for i in dollar_monthly.index:
            index = self.transformer_index[i]
            year = int(index[:4])
            month = int(index[-2:])

            date_1 = f"{year}-{month:02}-01"
            date_2 = f"{year}-{month:02}-29"
            dollar_monthly.loc[i, "azad"] = macro.dollar_azad.loc[date_1:date_2][
                "Close"
            ].mean()
            dollar_monthly.loc[i, "nima"] = macro.dollar_nima.loc[date_1:date_2][
                "Close"
            ].mean()
        dollar_monthly["ratio"] = dollar_monthly["nima"] / dollar_monthly["azad"]
        # converte_numeric(dollar_monthly)
        self.dollar_monthly = dollar_monthly
        ##### Create dollar rate ########
        (
            self.rate_dollar_azad_yearly,
            self.rate_dollar_nima_yearly,
        ) = self.create_dollar_rate(self.rate_yearly, "yearly")
        (
            self.rate_dollar_azad_quarterly,
            self.rate_dollar_nima_quarterly,
        ) = self.create_dollar_rate(self.rate_quarterly, "quarterly")
        (
            self.rate_dollar_azad_monthly,
            self.rate_dollar_nima_monthly,
        ) = self.create_dollar_rate(self.rate_monthly, "monthly")
        ##### Create dollar consump rate ########
        (
            self.rate_consump_dollar_azad_yearly,
            self.rate_consump_dollar_nima_yearly,
        ) = self.create_dollar_rate(self.rate_consump_yearly, "yearly")
        (
            self.rate_consump_dollar_azad_quarterly,
            self.rate_consump_dollar_nima_quarterly,
        ) = self.create_dollar_rate(self.rate_consump_quarterly, "quarterly")
        ##### Change dollar last month ########
        change_dollar_nima_monthly = (
            macro.dollar_nima.iloc[-1]["Close"] / dollar_monthly.iloc[-1]["nima"]
        )
        change_dollar_nima_quarterly = (
            self.dollar_nima.iloc[-1]["Close"] / dollar_quarterly.iloc[-1]["nima"]
        )
        dollar_nima_ratio = (
            self.dollar_nima.iloc[-1]["Close"] / self.dollar_azad.iloc[-1]["Close"]
        )
        self.change_dollar_nima_monthly = change_dollar_nima_monthly
        self.change_dollar_nima_quarterly = change_dollar_nima_quarterly
        self.dollar_nima_ratio = dollar_nima_ratio

        #######create shakhes #########
        self.shakhes = macro.shakhes_kol
        ####### create boors_kala data ############
        kala = macro.kala[["تولید کننده"]]
        lst = []
        try:
            for i in kala.index:
                if self.ful_name in kala.loc[i].values[0]:
                    lst.append(i)
            self.kala = macro.kala.loc[lst]
        except:
            pass

    def predict_value(self, pe_terminal=1):
        eps1 = self.pred_income.loc[self.future_year, "EPS_Capital"]
        eps2 = self.pred_income.loc[self.future_year + 1, "EPS_Capital"]
        dps_ratio = self.eps_data["ratio"].iloc[-1]
        dps1 = eps1 * dps_ratio
        dps2 = eps2 * dps_ratio
        self.dps_ratio = dps_ratio
        self.dps1 = dps1
        self.dps2 = dps2
        self.eps1 = eps1
        self.eps2 = eps2
        ## number of mounth to majma
        n = self.discounted_n
        value_d = eps1 / (1 + self.k) ** n + eps2 / (1 + self.k) ** (1 + n)
        value_d_dps = (dps1) / (1 + self.k) ** n + (dps2) / (1 + self.k) ** (1 + n)
        ### find majority goods
        df = self.price_revenue_yearly.iloc[[-1]].copy()
        df.drop(["total", "جمع"], axis=1, inplace=True)
        major_good = df.idxmax(axis=1).values[0]
        self.major_good = major_good

        ##### estimate eps of ngrowth year ######
        i = 3
        while i < 3 + self.n_g:
            vars()[f"eps{i}"] = (self.g_stock) * vars()[f"eps{i-1}"]
            vars()[f"dps{i}"] = (self.g_stock) * vars()[f"dps{i-1}"]
            # setattr(self,f"eps{i}",)
            self.__dict__[f"eps{i}"] = vars()[f"eps{i}"]
            self.__dict__[f"dps{i}"] = vars()[f"dps{i}"]
            value_d += vars()[f"eps{i}"] / (1 + self.k) ** (i - 1 + n)
            value_d_dps += vars()[f"dps{i}"] / (1 + self.k) ** (i - 1 + n)
            i += 1
        ##### Calculate terminal p/e ##########
        if pe_terminal == 1:
            pe_terminal_historical = self.pe_fw_historical[["pe"]].median()
            pe_terminal_historical = pe_terminal_historical.values[0]
            self.pe_terminal_historical = pe_terminal_historical
            pe_terminal_capm = (1 + self.g) / (self.k - self.g)
            self.pe_terminal_capm = pe_terminal_capm
            pe_terminal = np.average(
                [pe_terminal_historical, pe_terminal_capm], weights=[2, 1]
            )
            if pe_terminal > 9:
                pe_terminal = 9
        self.pe_terminal = pe_terminal
        ###### Calculate terminal value #######
        terminal_value = (vars()[f"eps{i-1}"] * pe_terminal) / (
            (1 + self.k) ** (i - 2 + n)
        )
        self.terminal_value = terminal_value
        value = value_d + terminal_value
        value_dps = value_d_dps + terminal_value
        self.value_d = value_d
        self.value_dps = value_dps
        self.value_d_dps = value_d_dps
        self.value = value
        self.potential_value_g = (value / self.Price["Close"].iloc[-1]) - 1
        # create eps estimate dataframe
        lst = list(range(self.future_year, self.future_year + self.n_g + 2))
        df = pd.DataFrame(index=lst, columns=["EPS", "DPS"])

        df["EPS"].loc[self.future_year] = self.pred_income["EPS_Capital"].loc[
            self.future_year
        ]
        df["EPS"].loc[self.future_year + 1] = self.pred_income["EPS_Capital"].loc[
            self.future_year + 1
        ]
        df["DPS"].loc[self.future_year] = dps1
        df["DPS"].loc[self.future_year + 1] = dps2
        for j in range(3, self.n_g + 2 + 1):
            df["EPS"].loc[self.future_year + j - 1] = vars()[f"eps{j}"]
            df["DPS"].loc[self.future_year + j - 1] = vars()[f"dps{j}"]
        self.eps_estimate = df

    def search_df_quarterly_monthly(self, df_y, df_q, df_m):
        dic_done = {}
        done = df_y.copy()
        for i in df_y.columns:
            q = search_df_quarterly(df_q, self.future_year, i)[0]
            m = search_df_quarterly(df_q, self.future_year, i)[1]
            if m != 0:
                month = self.fiscal_dic[self.fiscal_year][m] + 1
            else:
                month = 0
            if month < 10:
                id = f"{self.future_year}/0{month}"
            if month >= 10:
                id = f"{self.future_year}/{month}"
            if m != 0:
                try:
                    if self.industry != "palayesh":
                        dic_done[i] = q + df_m.loc[id:][i].sum()
                        done[i].loc[self.future_year] = q + df_m.loc[id:][i].sum()
                    if self.industry == "palayesh":
                        dic_done[i] = q
                        done[i].loc[self.future_year] = q
                except:
                    done[i].loc[self.future_year] = q
            else:
                done[i].loc[self.future_year] = 0
        return done

    # analyse detail trades
    def analyse_detail_trade(self, date):
        # read raw data
        df = pd.read_excel(
            f"{INDUSPATH}/{self.industry}/{self.Name}/{structure['detail']}{date}.xlsx"
        )
        # select desired data
        data = df[["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5"]]
        # preprocess data
        data.rename(
            columns={
                "Unnamed: 2": "n",
                "Unnamed: 3": "time",
                "Unnamed: 4": "volume",
                "Unnamed: 5": "price",
            },
            inplace=True,
        )
        data.drop(0, inplace=True)
        data["value"] = (data["price"] * data["volume"]) / 10**7
        # same time
        a = data[data["time"].duplicated(keep=False)]
        # change same time
        b = a["time"].duplicated()
        count = 0
        temp_first = []
        temp_last = []
        for i in range(len(b.index)):
            if b.loc[b.index[i]] == False:
                count += 1
                if count == 1:
                    temp_first.append(b.index[i])
                if count == 2:
                    temp_last.append(b.index[i - 1])
                    count = 0
        while len(temp_first) > len(temp_last):
            temp_first.pop(-1)
        volume = []
        ch_price = []
        value = []
        number = []
        time = []
        price = []
        # fill desired data
        for i in range(len(temp_first)):
            volume.append(a.loc[temp_first[i] : temp_last[i]]["volume"].sum())
            value.append(a.loc[temp_first[i] : temp_last[i]]["value"].sum())
            ch_price.append(
                a.loc[temp_last[i]]["price"] / a.loc[temp_first[i]]["price"]
            )
            number.append(len(a.loc[temp_first[i] : temp_last[i]]))
            time.append(a.loc[temp_first[i]]["time"])
            price.append(a.loc[temp_first[i]]["price"])
        process_detail = pd.DataFrame(columns=["volume", "price"])
        process_detail["volume"] = volume
        process_detail["price_ch"] = ch_price
        process_detail["value"] = value
        process_detail["number"] = number
        process_detail["time"] = time
        process_detail["price"] = price
        # send data to self
        self.detail_data = data
        self.same_data = a
        self.process_detail = process_detail
        return process_detail

    def pre_process_product_data(self):
        # rename_columns_monthly , yearly to same name
        rename_columns_dfs(self.count_revenue_monthly, self.count_revenue_yearly)
        rename_columns_dfs(self.price_revenue_monthly, self.price_revenue_yearly)
        rename_columns_dfs(self.price_revenue_quarterly, self.price_revenue_yearly)
        rename_columns_dfs(
            self.count_revenue_quarterly,
            self.count_revenue_yearly,
        )
        merge_same_columns(self.count_revenue_monthly)
        merge_same_columns(self.price_revenue_monthly)
        merge_same_columns(self.count_revenue_yearly)
        merge_same_columns(self.price_revenue_yearly)
        merge_same_columns(self.count_revenue_quarterly)
        merge_same_columns(self.price_revenue_quarterly)
        drop_non_same_columns(self.count_revenue_monthly, self.count_revenue_quarterly)
        drop_non_same_columns(self.price_revenue_monthly, self.price_revenue_quarterly)
        # create rate
        self.rate_monthly = self.price_revenue_monthly / self.count_revenue_monthly
        self.rate_monthly.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.rate_monthly.fillna(method="ffill", inplace=True)
        self.rate_monthly.fillna(method="bfill", inplace=True)
        self.rate_yearly = self.price_revenue_yearly / self.count_revenue_yearly
        self.rate_yearly.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.rate_yearly.fillna(method="ffill", inplace=True)
        self.rate_yearly.fillna(method="bfill", inplace=True)
        self.rate_quarterly = (
            self.price_revenue_quarterly / self.count_revenue_quarterly
        )
        self.rate_quarterly.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.rate_quarterly.fillna(method="ffill", inplace=True)
        self.rate_quarterly.fillna(method="bfill", inplace=True)

    def plot_price_value(self):
        pe_2 = (
            self.Price["Close"].iloc[-1]
            / self.pred_income.loc[self.future_year + 1, "EPS_Capital"]
        )
        pe_1 = (
            self.Price["Close"].iloc[-1]
            / self.pred_income.loc[self.future_year, "EPS_Capital"]
        )
        plt.figure(figsize=[20, 14])
        plt.subplot(2, 1, 1)
        plt.plot(self.Price["Close"])
        plt.axhline(self.value, linestyle="dashed", color="red")
        plt.title(self.Name)
        plt.subplot(2, 1, 2)
        plt.hist(self.pe_forward_adjust, edgecolor="black", bins=50)
        plt.axvline(pe_1, linestyle="dashed", color="red")
        plt.axvline(pe_2, linestyle="dashed", color="red", alpha=0.5)
        plt.title(self.Name)

    def create_end_data(self):
        end_data = self.income_yearly[["EPS_Capital"]]
        price = []
        price_first = []
        price_last = []
        min_price = []
        max_price = []
        price_ret = []
        pe_fw_yearly = []
        eps_yearly = []
        days = []
        vol = []
        lst_adjust_eps = []

        end_data["EPS_Capital"] = end_data["EPS_Capital"].apply(
            lambda x: 0.1 if x == 0 else x
        )
        pe_forward_adjust = {}
        for i in end_data.index:
            date_1 = f"{i}-01-01"
            date_2 = f"{i}-12-29"
            date_m1 = convert_to_miladi(date_1)
            date_m2 = convert_to_miladi(date_2)
            for j in pd.date_range(date_m1, date_m2):
                dt = date_m2 - j
                dt = dt.days

                try:
                    adjust_eps = end_data.loc[i]["EPS_Capital"] / (
                        (1 + self.k) ** (dt / 365)
                    )
                    pe_forward_adjust[convert_to_jalali(j)] = (
                        self.Price.loc[convert_to_jalali(j)]["Close"] / adjust_eps
                    )
                    days.append(dt)
                    eps_yearly.append(end_data.loc[i]["EPS_Capital"])
                    lst_adjust_eps.append(adjust_eps)
                except:
                    pass

            price.append(self.Price.loc[date_1:date_2]["Close"].mean())
            vol.append(self.Price.loc[date_1:date_2]["Value"].mean())
            min_price.append(self.Price.loc[date_1:date_2]["Close"].min())
            max_price.append(self.Price.loc[date_1:date_2]["Close"].max())
            pe_fw_yearly.append(
                self.Price.loc[date_1:date_2]["Close"].values
                / end_data["EPS_Capital"].loc[i]
            )
            try:
                price_first.append(self.Price.loc[date_1:date_2]["Close"][0])
            except:
                price_first.append(np.nan)
            try:
                price_last.append(self.Price.loc[date_1:date_2]["Close"][-1])
            except:
                price_last.append(np.nan)
        pe_forward_adjust = pd.DataFrame([pe_forward_adjust]).T
        pe_forward_adjust.rename(columns={0: "pe_adjust"}, inplace=True)
        pe_forward_adjust["eps"] = eps_yearly
        pe_forward_adjust["days"] = days
        pe_forward_adjust["eps_adjust"] = lst_adjust_eps
        pe_forward_adjust["price"] = self.Price["Close"]
        pe_forward_adjust["pe"] = pe_forward_adjust["price"] / pe_forward_adjust["eps"]
        pe_forward_adjust["ret"] = pe_forward_adjust["price"].pct_change()
        pe_forward_adjust["value"] = (
            pe_forward_adjust["eps_adjust"] * pe_forward_adjust["pe_adjust"].median()
        )
        pe_forward_adjust["value_trades"] = self.Price["Value"]
        self.pe_forward_adjust = pe_forward_adjust
        end_data["price"] = price
        end_data["max_price"] = max_price
        end_data["min_price"] = min_price
        end_data["first_price"] = price_first
        end_data["last_price"] = price_last
        end_data["volume"] = vol
        end_data["mean_price/eps"] = end_data["price"] / end_data["EPS_Capital"]
        end_data["max_price/eps"] = end_data["max_price"] / end_data["EPS_Capital"]
        end_data["min_price/eps"] = end_data["min_price"] / end_data["EPS_Capital"]
        end_data["first_price/eps"] = end_data["first_price"] / end_data["EPS_Capital"]
        end_data["last_price/eps"] = end_data["last_price"] / end_data["EPS_Capital"]
        end_data["volatility"] = (end_data["max_price"] / end_data["min_price"]) - 1
        end_data["yearly_ret"] = (end_data["last_price"] / end_data["first_price"]) - 1
        end_data["eps_ret"] = end_data["EPS_Capital"].pct_change()
        end_data["vol_ret"] = end_data["volume"].pct_change()
        ## predict future year+1
        try:
            end_data["min_price"].loc[self.future_year + 1] = (
                end_data["min_price/eps"].median()
                * end_data.loc[self.future_year + 1]["EPS_Capital"]
            )
            end_data["max_price"].loc[self.future_year + 1] = (
                end_data["max_price/eps"].median()
                * end_data.loc[self.future_year + 1]["EPS_Capital"]
            )
            end_data["price"].loc[self.future_year + 1] = (
                end_data["mean_price/eps"].median()
                * end_data.loc[self.future_year + 1]["EPS_Capital"]
            )
        except:
            pass
        self.end_data = end_data
        self.pe_fw_yearly = pe_fw_yearly
        self.eps_yearly = eps_yearly
        pe_fw_historical = []
        for i in self.pe_fw_yearly:
            pe_fw_historical.extend(i.tolist())
        self.pe_fw_historical = pd.DataFrame(np.array(pe_fw_historical))
        date_1 = f"{end_data.index[0]}-01-01"
        self.pe_fw_historical.set_index(self.Price[date_1:].index, inplace=True)
        self.pe_fw_historical["price"] = self.Price["Close"]
        self.pe_fw_historical["cret"] = (
            self.pe_fw_historical["price"] / self.pe_fw_historical["price"].iloc[0]
        )
        self.pe_fw_historical.rename(columns={0: "pe"}, inplace=True)
        self.pe_fw_historical["eps"] = (
            self.pe_fw_historical["price"] / self.pe_fw_historical["pe"]
        )
        self.days = days
        self.lst_adjust_eps = lst_adjust_eps

    def box_plot(self):
        plt.figure(figsize=[15, 8])
        plt.subplot(1, 2, 1)
        self.pe_fw_historical["pe"].plot(kind="box")
        plt.axhline(self.pe_fw_historical["pe"].iloc[-1], linestyle="dashed")
        plt.subplot(1, 2, 2)
        self.pred_com["سود (زیان) خالص"].plot(kind="box")
        plt.axhline(self.pred_com["سود (زیان) خالص"].iloc[-1], linestyle="dashed")

    def plot_end_data(self, year, year_eps):
        t1 = pd.to_datetime(JalaliDate(year, 1, 1).to_gregorian())
        t2 = pd.to_datetime(JalaliDate(year, 12, 29).to_gregorian())
        pe = {}
        price = []
        lst_adjust_eps = []
        for i in pd.date_range(t1, t2):
            dt = t2 - i
            dt = dt.days
            try:
                adjust_eps = self.end_data["EPS_Capital"].loc[year_eps] / (
                    1 + self.k
                ) ** ((year_eps - year) + dt / 365)

                pe[i] = self.Price.loc[i]["Close"] / adjust_eps
                lst_adjust_eps.append(adjust_eps)
                price.append(self.Price.loc[i]["Close"])
            except:
                pass
        pe = pd.DataFrame([pe]).T
        pe.rename(columns={0: "pe"}, inplace=True)
        pe["price"] = price
        pe["ret"] = pe["price"].pct_change()
        pe["cret"] = pe["price"] / pe["price"].iloc[0]
        pe["adjust_eps"] = lst_adjust_eps
        plt.figure(figsize=[20, 12])
        plt.subplot(2, 2, 1)
        plt.plot(pe["pe"])
        plt.title("pe")
        plt.subplot(2, 2, 2)
        plt.plot(pe["cret"])
        plt.title("Cret")
        plt.subplot(2, 2, 3)
        plt.hist(pe["pe"], edgecolor="black")
        plt.axvline(pe["pe"].iloc[-1], color="red")
        plt.axvline(pe["pe"].median(), linestyle="dashed")
        plt.title("pe")
        plt.subplot(2, 2, 4)
        plt.hist(pe["ret"], edgecolor="black")
        plt.axvline(pe["ret"].iloc[-1], color="red")
        plt.axvline(pe["ret"].median(), linestyle="dashed")
        plt.title("ret")
        plt.figure(figsize=[20, 8])
        plt.subplot(2, 1, 1)
        plt.plot(self.pe_forward_adjust["pe"])
        plt.title("PE_Forward")
        plt.subplot(2, 1, 2)
        plt.hist(self.pe_forward_adjust["pe"], edgecolor="black", bins=40)
        plt.axvline(self.pe_forward_adjust["pe"].iloc[-1], color="red")
        plt.axvline(self.pe_forward_adjust["pe"].median(), linestyle="dashed")

        return pe

    def create_cumulative_data(self, type_user, year=1401, q=3, m2=11):
        """
        type_user=count,price,income,cost,count_consump,price_consump
        """
        lst = []
        if type_user == "count":
            data_q = self.count_revenue_quarterly.copy()
            data_m = self.count_revenue_monthly.copy()

        if type_user == "price":
            data_q = self.price_revenue_quarterly.copy()
            data_m = self.price_revenue_monthly.copy()
        if type_user == "income":
            data_q = self.income_quarterly.copy()
        if type_user == "cost":
            data_q = self.cost_quarterly.copy()
        if type_user == "overhead":
            data_q = self.overhead_quarterly.copy()
        if type_user == "count_consump":
            data_q = self.count_consump_quarterly.copy()
        if type_user == "price_consump":
            data_q = self.price_consump_quarterly.copy()
        if (
            (type_user == "count")
            or (type_user == "price")
            and (self.industry != "palayesh")
        ):
            lst = []
            for i in data_q.index:
                if (int(i[:4]) == year) and (int(i[-1]) <= q):
                    lst.append(i)
            if len(lst) != 0:
                df_cum_q = data_q.loc[lst].cumsum().iloc[[-1]]
            else:
                df_cum_q = 0
            m1 = q * 3 + 1
            if m1 > 12:
                m1 = m1 % 12
            id_m1 = f"{year}/{m1}"
            id_m2 = f"{year}/{m2}"
            if m2 >= m1:
                df_cum_m = data_m.loc[id_m1:id_m2].cumsum().iloc[[-1]]
                if len(lst) != 0:
                    df_cum_m.rename(
                        index={df_cum_m.index[0]: df_cum_q.index[0]}, inplace=True
                    )
                    df = df_cum_m + df_cum_q
                else:
                    df = df_cum_m
            else:
                df = df_cum_q
        if ((type_user == "count") or (type_user == "price")) and (
            self.industry == "palayesh"
        ):
            lst = []
            for i in data_q.index:
                if (int(i[:4]) == year) and (int(i[-1]) <= q):
                    lst.append(i)
            if len(lst) != 0:
                df_cum_q = data_q.loc[lst].cumsum().iloc[[-1]]
            else:
                df_cum_q = 0
            df = df_cum_q

        if (
            (type_user == "cost")
            or (type_user == "income")
            or (type_user == "count_consump")
            or (type_user == "price_consump")
            or (type_user == "overhead")
        ):
            lst = []
            for i in data_q.index:
                if (int(i[:4]) == year) and (int(i[-1]) <= q):
                    lst.append(i)
            if len(lst) != 0:
                if (
                    (type_user == "cost")
                    or (type_user == "income")
                    or (type_user == "overhead")
                ):
                    df = data_q.loc[lst].cumsum().iloc[[-1]]
                if (type_user == "count_consump") or (type_user == "price_consump"):
                    for i in data_q.loc[lst].index:
                        for j in data_q.loc[lst].columns:
                            data_q.loc[i, j] = (
                                data_q.loc[i, j] * self.inventory_ratio_quarterly.loc[i]
                            )
                    df = data_q.loc[lst].cumsum().iloc[[-1]]
            else:
                df = 0

        if isinstance(df, pd.DataFrame):
            df.rename(index={df.index[0]: year}, inplace=True)
            df.fillna(0, inplace=True)
        return df

    def reindex_monthly_data(self, user_type="count"):
        if user_type == "count":
            df = self.count_revenue_monthly
        if user_type == "price":
            df = self.price_revenue_monthly
        new_id = []
        for i in df.index:
            y = int(i[:4])
            m = int(i[-2:])
            if m > self.fiscal_year:
                new_y = y + 1
                res = m % self.fiscal_year
                mul = np.floor(m / self.fiscal_year)
                new_m = int((mul - 1) * self.fiscal_year + res)
            if m <= self.fiscal_year:
                new_m = (12 - self.fiscal_year) + m
                new_y = y
            id = f"{new_y}/{new_m}"
            new_id.append(id)
        df["new_id"] = new_id
        last_id = df.index.values
        transformer = dict(zip(new_id, last_id))
        self.transformer_index = transformer
        df.set_index("new_id", inplace=True)

    def create_dollar_rate(self, df, time_type):
        """
        time_type : yearly,quarterly,monthly
        return : azad and nima rate
        """
        time_dic = {
            "yearly": self.dollar_yearly,
            "quarterly": self.dollar_quarterly,
            "monthly": self.dollar_monthly,
        }
        rate_dollar_azad = pd.DataFrame(index=df.index, columns=df.columns)
        rate_dollar_nima = pd.DataFrame(index=df.index, columns=df.columns)
        for i in rate_dollar_azad.columns:
            rate_dollar_azad[i] = (df[i] / time_dic[time_type]["azad"]) * 1000000
            rate_dollar_nima[i] = (df[i] / time_dic[time_type]["nima"]) * 1000000
        return rate_dollar_azad, rate_dollar_nima

    def predict_cost(
        self,
    ):
        df_cost = self.cost_yearly.iloc[[-1]].copy()
        pred_cost = CostDataFrame(df_cost, self.pred_overhead)
        pred_cost.loc[self.future_year] = 0
        pred_cost.loc[self.future_year + 1] = 0
        pred_cost.loc[self.future_year, "مواد مستقیم مصرفی"] = self.pred_material.loc[
            self.future_year
        ].values[0]
        pred_cost.loc[
            self.future_year + 1, "مواد مستقیم مصرفی"
        ] = self.pred_material.loc[self.future_year + 1].values[0]
        pred_cost.loc[
            self.future_year, "دستمزد مستقیم تولید"
        ] = self.pred_direct_salary.loc[self.future_year].values[0]
        pred_cost.loc[
            self.future_year + 1, "دستمزد مستقیم تولید"
        ] = self.pred_direct_salary.loc[self.future_year + 1].values[0]
        pred_cost.update_dependent_columns()

        self.pred_cost = pred_cost

    def predict_revenue(self):
        ########################################### calculate Revenue #########################################
        price_revenue_residual = self.count_revenue_residual * self.rate_residual
        price_revenue_residual["total"] = price_revenue_residual.sum(axis=1)
        self.price_revenue_residual = price_revenue_residual
        add_extra_columns_dfs(self.count_revenue_residual, self.count_revenue_done)
        add_extra_columns_dfs(self.price_revenue_residual, self.price_revenue_done)

        self.count_revenue_done.loc[self.future_year + 1] = 0
        self.price_revenue_done.loc[self.future_year + 1] = 0
        pred_count_revenue = self.count_revenue_residual + self.count_revenue_done
        pred_price_revenue = price_revenue_residual + self.price_revenue_done
        self.pred_price_revenue = pred_price_revenue

        self.pred_revenue = pred_price_revenue[["total"]]
        self.pred_count_revenue = pred_count_revenue

    def predict_rate_consump(self):
        df = self.rate_consump_quarterly.copy()
        df.drop(["جمع"], axis=1, inplace=True)
        consump_lst = df.columns.values
        self.consum_lst = consump_lst
        q_residual = 4 - self.last_q
        pred_rate_consump = pd.DataFrame(
            columns=consump_lst, index=[self.future_year, self.future_year + 1]
        )
        for i in consump_lst:
            model = pm.auto_arima(df[i], seasonal=True, m=4)
            vars()[f"model_consump{i}"] = model
            self.__dict__[f"model_consump{i}"] = vars()[f"model_consump{i}"]
            rate_fu = model.predict(q_residual)
            rate_fu = rate_fu.mean()
            pred_rate_consump.loc[self.future_year, i] = rate_fu
            rate_next = model.predict(q_residual + 4)
            rate_next = rate_next[-4:].mean()
            pred_rate_consump.loc[self.future_year + 1, i] = rate_next
        self.pred_rate_consump = pred_rate_consump

    def predict_count_consump(self):
        income_residual = pd.DataFrame(
            columns=["revenue"], index=[self.future_year, self.future_year + 1]
        )
        #### create_count_done #####
        count_quarter_done = pd.DataFrame(
            columns=self.products_lst, index=[self.future_year, self.future_year + 1]
        )
        for i in self.products_lst_q:
            count_quarter_done.loc[self.future_year, i] = search_df_quarterly(
                self.count_revenue_quarterly, self.future_year, i
            )[0]
        count_quarter_done.loc[self.future_year + 1] = 0
        self.count_quarter_done = count_quarter_done
        count_quarter_residual = self.pred_count_revenue - self.count_quarter_done
        count_quarter_residual.drop("total", axis=1, inplace=True)
        self.count_quarter_residual = count_quarter_residual
        count_consump_done = self.create_cumulative_data(
            "count_consump", self.future_year, self.last_q
        )
        self.count_consump_done = count_consump_done
        if isinstance(self.income_done, pd.DataFrame):
            income_residual.loc[self.future_year] = np.squeeze(
                self.pred_revenue.loc[self.future_year].values
                - self.income_done["فروش"].values
            )
            income_residual.loc[self.future_year + 1] = np.squeeze(
                self.pred_revenue.loc[self.future_year + 1].values
            )
        else:
            income_residual.loc[self.future_year] = self.pred_revenue.loc[
                self.future_year
            ].values[0]
            income_residual.loc[self.future_year + 1] = self.pred_revenue.loc[
                self.future_year + 1
            ].values[0]
        self.income_residual = income_residual
        material_residual = pd.DataFrame(
            columns=["material"], index=[self.future_year, self.future_year + 1]
        )

        ###### create_material_data ########
        df = self.count_consump_quarterly.copy()
        df.drop(["جمع"], axis=1, inplace=True)
        consump_lst = df.columns.values
        self.consump_lst = consump_lst
        consump_col = [i + "consump" for i in df.columns]
        df.columns = consump_col
        df = pd.concat([df, self.count_revenue_quarterly], axis=1)
        df.drop(["جمع", "total"], axis=1, inplace=True)
        df.dropna(inplace=True)
        self.material_data = df
        ##### Create_model material ########
        y = df[consump_col]
        x = df.drop(consump_col, axis=1)
        x = sm.add_constant(x)
        model_count_consump = sm.OLS(y, x).fit()
        self.model_count_consump = model_count_consump
        #### evaluate model ########
        pred = model_count_consump.predict(x)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame()
        pred.columns = y.columns
        dic = {}
        for i in y.columns:
            dic[i] = r2_score(pred[i], y[i])
        score = pd.DataFrame(dic, index=[0])
        score.columns = consump_lst
        self.score_consump = score
        self.dic_score = dic
        x_count_revenue_residual = count_quarter_residual.copy()
        x_count_revenue_residual = sm.add_constant(x_count_revenue_residual)
        count_consump_residual = model_count_consump.predict(x_count_revenue_residual)
        if isinstance(count_consump_residual, pd.Series):
            count_consump_residual = count_consump_residual.to_frame()
        count_consump_residual.columns = consump_lst
        ##### fill negative prediction #######
        if isinstance(count_consump_residual, pd.DataFrame):
            count_consump_residual = count_consump_residual.applymap(
                lambda x: 0 if x < 0 else x
            )

        #### fill bad data ######
        bad_consump = []
        for i in count_consump_residual.columns:
            if score[i].values < 0.5:
                bad_consump.append(i)
        self.bad_consump = bad_consump
        for i in bad_consump:
            try:
                count_consump_residual.loc[self.future_year, i] = (
                    self.count_consump_yearly.iloc[-1][i] - count_consump_done[i]
                ).values[0]
                count_consump_residual.loc[
                    self.future_year + 1, i
                ] = self.count_consump_yearly.iloc[-1][i]

            except:
                count_consump_residual.loc[self.future_year, i] = (
                    self.create_cumulative_data(
                        "count_consump", self.future_year - 1, 4
                    )[i].values[0]
                    - count_consump_done[i].values[0]
                )
                count_consump_residual.loc[
                    self.future_year + 1, i
                ] = self.create_cumulative_data(
                    "count_consump", self.future_year - 1, 4
                )[
                    i
                ].values[
                    0
                ]
        #### fill zero data with last year ######
        for i in count_consump_residual.index:
            for j in count_consump_residual.columns:
                if count_consump_residual.loc[i, j] == 0:
                    try:
                        count_consump_residual.loc[i, j] = (
                            self.count_consump_yearly.iloc[[-1]][j].values
                            - count_consump_done[j].values
                        )
                    except:
                        pass

        self.count_consump_residual = count_consump_residual

    def predict_material(self):
        price_consump_residual = self.pred_rate_consump * self.count_consump_residual
        price_consump_residual["total"] = price_consump_residual.sum(axis=1)
        self.price_consump_residual = price_consump_residual
        pred_material = pd.DataFrame(
            columns=["material"], index=[self.future_year, self.future_year + 1]
        )
        if self.material == "default":
            pred_material.loc[self.future_year] = (
                self.cost_done["مواد مستقیم مصرفی"].values[0]
                + self.price_consump_residual["total"].loc[[self.future_year]].values[0]
            )
            pred_material.loc[self.future_year + 1] = (
                +self.price_consump_residual["total"]
                .loc[[self.future_year + 1]]
                .values[0]
            )
            self.pred_material = pred_material

    def predict_non_direct_material(self):
        rev_mat = pd.DataFrame(columns=["revenue", "material"])
        rev_mat["revenue"] = self.income_quarterly["فروش"]
        rev_mat["material"] = self.cost_quarterly["مواد مستقیم مصرفی"]
        rev_mat.dropna(inplace=True)
        self.rev_mat = rev_mat
        y = rev_mat["material"]
        x = rev_mat[["revenue"]]
        model = linear.LinearRegression()
        model.fit(x, y)
        self.model_rev_mat = model
        self.pred_material.loc[self.future_year] = (
            model.predict(self.income_residual)[0]
            + self.cost_done["مواد مستقیم مصرفی"].values[0]
        )
        self.pred_material = pd.DataFrame(
            columns=["material"], index=[self.future_year, self.future_year + 1]
        )
        self.pred_material.loc[self.future_year + 1] = model.predict(
            self.income_residual
        )[1]
        self.predict_material_over()
        self.predict_overhead()
        self.predict_cost()
        self.predict_income()

    def predict_overhead(
        self,
    ):
        df_overhead = self.overhead_yearly.iloc[[-1]].copy()
        pred_overhead = OverheadDataFrame(df_overhead)
        pred_overhead.loc[self.future_year] = 0
        pred_overhead.loc[self.future_year + 1] = 0
        pred_overhead.loc[
            self.future_year, "هزینه حقوق و دستمزد"
        ] = self.pred_overhead_salary.loc[self.future_year].values[0]
        pred_overhead.loc[
            self.future_year + 1, "هزینه حقوق و دستمزد"
        ] = self.pred_overhead_salary.loc[self.future_year + 1].values[0]
        pred_overhead.loc[
            self.future_year, "هزینه انرژی (آب، برق، گاز و سوخت)"
        ] = self.pred_energy.loc[self.future_year].values[0]
        pred_overhead.loc[
            self.future_year + 1, "هزینه انرژی (آب، برق، گاز و سوخت)"
        ] = self.pred_energy.loc[self.future_year + 1].values[0]
        if "هزینه حمل و نقل و انتقال" in pred_overhead.columns:
            pred_overhead.loc[
                self.future_year, "هزینه حمل و نقل و انتقال"
            ] = self.pred_transport.loc[self.future_year].values[0]
            pred_overhead.loc[
                self.future_year + 1, "هزینه حمل و نقل و انتقال"
            ] = self.pred_transport.loc[self.future_year + 1].values[0]
        if "سایر هزینه ها" in pred_overhead.columns:
            pred_overhead.loc[
                self.future_year, "سایر هزینه ها"
            ] = self.pred_other_over.loc[self.future_year].values[0]
            pred_overhead.loc[
                self.future_year + 1, "سایر هزینه ها"
            ] = self.pred_other_over.loc[self.future_year + 1].values[0]
        if "هزینه مواد مصرفی" in pred_overhead.columns:
            pred_overhead.loc[
                self.future_year, "هزینه مواد مصرفی"
            ] = self.pred_material_over.loc[self.future_year].values[0]
        if "هزینه مواد مصرفی" in pred_overhead.columns:
            pred_overhead.loc[
                self.future_year + 1, "هزینه مواد مصرفی"
            ] = self.pred_material_over.loc[self.future_year + 1].values[0]
        pred_overhead.update_dependent_columns()
        self.pred_overhead = pred_overhead

    def predict_parameter(
        self,
    ):
        ####### create required data #######
        future_year = self.income_yearly.index[-1] + 1
        self.future_year = future_year
        q = []
        for i in self.count_revenue_quarterly.index:
            if int(i[:4]) == self.future_year:
                q.append(int(i[-1]))
            if int(i[-1]) == 4:
                last_q = 4
        if len(q) != 0:
            last_q = max(q)
        else:
            last_q = 0
        # last_q = int(self.count_revenue_quarterly.index[-1][-1])
        self.last_q = last_q
        last_m = int(self.count_revenue_monthly.iloc[[-1]].index[0].split("/")[1])
        year_m = int(self.count_revenue_monthly.iloc[[-1]].index[0].split("/")[0])
        self.year_m = year_m
        if year_m > future_year:
            last_m = 12
        self.last_m = last_m
        ### find majority goods
        df = self.price_revenue_yearly.iloc[[-1]].copy()
        df.drop(["total", "جمع"], axis=1, inplace=True)
        major_good = df.idxmax(axis=1).values[0]
        self.major_good = major_good
        df2 = self.price_consump_yearly.iloc[[-1]].copy()
        df2.drop(["جمع"], axis=1, inplace=True)
        major_consump = df2.idxmax(axis=1).values[0]
        self.major_consump = major_consump
        self.future_year = future_year
        overhead_done = self.create_cumulative_data(
            "overhead", year=self.future_year, q=self.last_q
        )
        overhead_last = self.create_cumulative_data(
            "overhead", year=self.future_year - 1, q=self.last_q
        )
        self.overhead_done = overhead_done
        self.overhead_last = overhead_last
        self.cost_done = self.create_cumulative_data(
            "cost", year=self.future_year, q=self.last_q
        )
        self.income_done = self.create_cumulative_data(
            "income", year=self.future_year, q=self.last_q
        )
        if isinstance(overhead_done, pd.DataFrame):
            overhead_cum = pd.concat([overhead_last, overhead_done])
        else:
            overhead_cum = 0
        self.overhead_cum = overhead_cum
        cost_done = self.create_cumulative_data(
            "cost", year=self.future_year, q=self.last_q
        )
        cost_last = self.create_cumulative_data(
            "cost", year=self.future_year - 1, q=self.last_q
        )
        self.cost_done = cost_done
        if isinstance(cost_done, pd.DataFrame):
            cost_cum = pd.concat([cost_last, cost_done])
        else:
            cost_cum = 0
        self.cost_cum = cost_cum
        if isinstance(overhead_cum, pd.DataFrame):
            overhead_cum_ch = overhead_cum.pct_change()
        else:
            overhead_cum_ch = 0
        self.overhead_cum_ch = overhead_cum_ch
        if isinstance(cost_cum, pd.DataFrame):
            cost_cum_ch = cost_cum.pct_change()
        else:
            cost_cum_ch = 0
        self.cost_cum_ch = cost_cum_ch
        ##### predict alpha and material parameters #######
        #### predict_alpha_rate ######
        ### find majority goods
        df = self.price_revenue_yearly.iloc[[-1]].copy()
        df.drop(["total", "جمع"], axis=1, inplace=True)
        major_good = df.idxmax(axis=1).values[0]
        self.major_good = major_good
        ### cumulative calculate from quarterly #############
        # add_extra_columns_dfs(self.count_revenue_quarterly, self.count_revenue_yearly)
        # add_extra_columns_dfs(self.count_revenue_monthly, self.count_revenue_yearly)
        last_count_revenue = self.count_revenue_yearly.iloc[[-1]].copy()
        last_count_revenue.drop(["جمع", "total"], axis=1, inplace=True)
        self.last_count_revenue = last_count_revenue
        df_cum_last_year = self.create_cumulative_data(
            type_user="count", year=self.future_year - 1, q=self.last_q, m2=self.last_m
        )
        df_cum_future_year = self.create_cumulative_data(
            type_user="count", year=self.future_year, q=self.last_q, m2=self.last_m
        )

        if isinstance(df_cum_future_year, pd.DataFrame):
            # calculate count rev done

            self.count_revenue_done = self.create_cumulative_data(
                "count", self.future_year, self.last_q, self.last_m
            )
            self.price_revenue_done = self.create_cumulative_data(
                "price", self.future_year, self.last_q, self.last_m
            )
            self.count_revenue_done.drop("جمع", axis=1, inplace=True)
            self.price_revenue_done.drop("جمع", axis=1, inplace=True)
            df_cum_count = pd.concat([df_cum_last_year, df_cum_future_year])
            self.df_cum_count = df_cum_count

        ###### products lst ##########
        df = self.count_revenue_monthly.copy()
        df.drop(["total", "جمع"], axis=1, inplace=True)
        products_lst = df.columns.values
        self.products_lst = products_lst
        df_q = self.count_revenue_quarterly.copy()
        df_q.drop(["total", "جمع"], axis=1, inplace=True)
        products_lst_q = df_q.columns.values
        self.products_lst_q = products_lst_q
        #################### Predict_value_parameters ############3######
        ## calculate historical expected_return
        delta = convert_to_miladi(self.Price["Close"].index[-1]) - convert_to_miladi(
            self.Price["Close"].index[0]
        )
        years = delta.days / 365
        self.years = years
        re_historical = (
            self.Price["Close"].iloc[-1] / self.Price["Close"].iloc[0]
        ) ** (1 / years)
        self.re_historical = re_historical
        ### calculate Beta ####
        data_shakhes = pd.concat([self.Price["Close"], self.shakhes["Close"]], axis=1)
        data_shakhes.columns = ["stock", "shakhes"]
        data_shakhes.dropna(inplace=True)
        data_shakhes = np.log(data_shakhes / data_shakhes.shift(1))
        data_shakhes.dropna(inplace=True)
        cov = data_shakhes.cov().iloc[0, 1]
        var = data_shakhes["shakhes"].var()
        beta = cov / var
        self.beta = beta
        self.data_shakhes = data_shakhes
        ##### calculate expected_return #####
        k_capm = self.rf + self.beta * self.erp
        k_historical = self.re_historical - 1
        self.k_historical = k_historical
        self.k_capm = k_capm
        ### calculate aggregate growth #######
        rate_aggr = (
            self.rate_yearly[major_good][-5:] / self.rate_yearly[major_good].iloc[-5]
        )
        cagr_rate = math.pow(
            (
                self.rate_yearly[major_good][-5:]
                / self.rate_yearly[major_good].iloc[-5]
            ).iloc[-1],
            1 / (5 - 1),
        )
        self.rate_aggr = rate_aggr
        self.cagr_rate = cagr_rate
        count_aggr = (
            self.count_revenue_yearly[major_good][-5:]
            / self.count_revenue_yearly[major_good].iloc[-5]
        )
        cagr_count = math.pow(count_aggr.iloc[-1], 1 / (len(count_aggr) - 1))
        self.count_aggr = count_aggr
        self.cagr_count = cagr_count
        profit_aggr = (
            self.income_yearly["سود (زیان) خالص"][-5:]
            / self.income_yearly["سود (زیان) خالص"].iloc[-5]
        )
        self.profit_aggr = profit_aggr
        try:
            cagr_profit = math.pow(profit_aggr.iloc[-1], 1 / (len(profit_aggr) - 1))
            self.cagr_profit = cagr_profit
        except Exception as err:
            self.error.append(f"cant calculate cagr profit : {err}")

        ###### Calculate G #######
        self.g_economy = 0.02 + self.rf
        self.g_stock = (cagr_count) * (cagr_rate)

        value_parameter = {
            "g_stock": self.g_stock,
            "cagr_rate": self.cagr_rate,
            "cagr_count": self.cagr_count,
        }
        self.value_parameter = value_parameter
        ### create rev_data_matrix ####
        df = pd.DataFrame(columns=["revenue"])
        df["revenue"] = self.income_quarterly["فروش"]
        df["material"] = self.cost_quarterly["مواد مستقیم مصرفی"]
        df["salary"] = (
            self.cost_quarterly["دستمزد مستقیم تولید"]
            + self.overhead_quarterly["هزینه حقوق و دستمزد"]
            + self.official_quarterly["هزینه حقوق و دستمزد"]
        )
        df["energy"] = self.overhead_quarterly["هزینه انرژی (آب، برق، گاز و سوخت)"]
        df.dropna(inplace=True)
        df["mat/rev"] = df["material"] / df["revenue"]
        df["energy/rev"] = df["energy"] / df["revenue"]
        df["salary/rev"] = df["salary"] / df["revenue"]
        self.rev_data_quarterly = df
        df = pd.DataFrame(columns=["revenue"])
        df["revenue"] = self.income_yearly["فروش"]
        df["material"] = self.cost_yearly["مواد مستقیم مصرفی"]
        df["salary"] = (
            self.cost_yearly["دستمزد مستقیم تولید"]
            + self.overhead_yearly["هزینه حقوق و دستمزد"]
            + self.official_yearly["هزینه حقوق و دستمزد"]
        )
        df["energy"] = self.overhead_yearly["هزینه انرژی (آب، برق، گاز و سوخت)"]
        df.dropna(inplace=True)
        df["mat/rev"] = df["material"] / df["revenue"]
        df["energy/rev"] = df["energy"] / df["revenue"]
        df["salary/rev"] = df["salary"] / df["revenue"]
        self.rev_data_yearly = df

    def predict_opex(self):
        df = pd.DataFrame(columns=["opex", "revenue"])
        df["opex"] = -self.income_quarterly["هزینه های عمومی, اداری و تشکیلاتی"]
        df["revenue"] = self.income_quarterly["فروش"]
        x = df[["revenue"]]
        y = df["opex"]
        model = linear.LinearRegression()
        model.fit(x, y)
        self.model_opex = model
        df["pred"] = model.predict(x)
        self.opex_data = df
        pred_opex = pd.DataFrame(
            columns=["opex"], index=[self.future_year, self.future_year + 1]
        )
        pred_opex.loc[self.future_year] = (
            model.predict(
                [[self.price_revenue_residual.loc[self.future_year, "total"]]]
            )
            - self.income_done["هزینه های عمومی, اداری و تشکیلاتی"].values[0]
        )
        pred_opex.loc[self.future_year + 1] = model.predict(
            [[self.price_revenue_residual.loc[self.future_year + 1, "total"]]]
        )
        self.pred_opex = pred_opex

    def predict_tax(self):
        rev_tax = pd.DataFrame(columns=["pretax", "tax"])
        rev_tax["pretax"] = self.income_yearly["Pretax_Income"]
        rev_tax["tax"] = -self.income_yearly["Tax_Provision"]
        rev_tax["ratio"] = rev_tax["tax"] / rev_tax["pretax"]
        model = linear.LinearRegression()
        model.fit(rev_tax[["pretax"]], rev_tax["tax"])
        self.model_rev_tax = model
        rev_tax["pred"] = model.predict(rev_tax[["pretax"]])
        rev_tax["error"] = rev_tax["tax"] - rev_tax["pred"]
        self.rev_tax = rev_tax
        tax_future = self.model_rev_tax.predict(
            self.pred_income[["Pretax_Income"]].iloc[-2:]
        )
        self.pred_income.loc[self.future_year, "Tax_Provision"] = -tax_future[0]
        self.pred_income.loc[self.future_year + 1, "Tax_Provision"] = -tax_future[1]

    def predict_other_over(self):
        df = pd.DataFrame(columns=["other", "revenue"])
        df["other"] = self.overhead_yearly["سایر هزینه ها"]
        df["revenue"] = self.income_yearly["فروش"]
        model = linear.LinearRegression()
        x = df[["revenue"]]
        y = df["other"]
        model.fit(x, y)
        df["pred"] = model.predict(x)
        self.other_over_data = df
        pred_other_over = pd.DataFrame(
            columns=["other"], index=[self.future_year, self.future_year + 1]
        )
        pred_other_over.loc[self.future_year] = (
            model.predict(
                [
                    [
                        self.pred_revenue.loc[self.future_year].values[0]
                        - self.income_done["فروش"].values[0]
                    ]
                ]
            )
            + self.overhead_done["سایر هزینه ها"].values[0]
        )
        pred_other_over.loc[self.future_year + 1] = model.predict(
            self.pred_revenue.loc[[self.future_year + 1]]
        )
        self.pred_other_over = pred_other_over

    def update_dfs(self):
        self.pred_overhead.update_dependent_columns()
        self.pred_cost.update_dependent_columns()
        self.pred_income.update_dependent_columns()
        self.create_end_data()
        self.create_fcfe()
        self.predict_value()

    def plot_rate(self):
        for i in self.rate_monthly.columns:
            try:
                plt.figure()
                self.rate_monthly[i].iloc[-15:].plot(kind="bar")
                reshaped_text = arabic_reshaper.reshape(i)
                persian_text = get_display(reshaped_text)
                plt.title(persian_text)
                plt.axhline(self.pred_rate[i].iloc[0], linestyle="dashed")
                plt.axhline(self.pred_rate[i].iloc[1], linestyle="dashed")
            except:
                pass

    def plot_structure(self):
        df = -self.income_yearly[
            [
                "Cost_of_Revenue",
                "Operating_Expense",
                "Interest_Expense",
                "Tax_Provision",
            ]
        ]
        df.iloc[-1].plot(kind="pie", autopct="%.2f")
        plt.title("structure of cost")
        plt.figure()
        df = self.income_yearly[
            [
                "Gross_Profit",
                "Other_non_operate_Income_Expense",
                "Other_operating_Income_Expense",
            ]
        ]
        df.iloc[-1].plot(kind="pie", autopct="%.2f")
        plt.title("structure of income")

    def value_monte_corlo_simulation(self):
        ##### cretae mont corlo probablity of rate residual #######
        # calculate number of residual month
        number_res_month = 12 - int(self.rate_monthly.index[-1].split("/")[1])
        # create mont carlo of major_good rate
        mont_carlo_rate = monte_carlo_simulate(
            self.rate_monthly[self.major_good], 100, number_res_month
        )
        mont_carlo_mean = np.mean(mont_carlo_rate)
        # calculate rate of other goods
        temp = self.pred_rate[self.major_good].values
        temp = temp.T
        temp = temp.reshape(2, -1)
        rate_com = self.pred_rate / temp
        rate_com
        ### predict income for multiple rates ###
        lst = []
        for i in mont_carlo_mean:
            self.pred_rate.loc[1402, self.major_good] = i
            rate_m = self.pred_rate[self.major_good].values
            rate_m = rate_m.reshape(2, -1)
            self.pred_rate = rate_com * rate_m
            self.predict_revenue()
            self.predict_income()
            lst.append(self.pred_income.loc[1402, "EPS_Capital"])
        eps = np.array(lst)
        value = self.pe_forward_adjust["pe"].median() * eps
        self.prob_value = value
        self.mont_carlo_mean = mont_carlo_mean
        self.mont_carlo_rate = mont_carlo_rate

    def ret_analyse(self):
        ret = self.Price[["Ret"]]
        idx = ret[ret["Ret"] == 0].index
        ret.drop(idx, inplace=True)
        self.ret = ret
        norm_dis = stats.norm(ret.mean(), ret.std())
        self.ret_norm_dis = norm_dis
        deg, loc, scale = stats.t.fit(ret)
        t_dis = stats.t(deg, loc, scale)
        self.ret_t_dis = t_dis

    def monte_carlo(self, n_sim=100, n_days=100):
        self.ret_analyse()
        last_price = self.Price.iloc[-1]["Close"]
        df = pd.DataFrame()
        for i in range(n_sim):
            count = 0
            prices = []
            prices.append(last_price)
            for d in range(n_days):
                ret = self.ret_norm_dis.rvs()
                price = prices[count] * (1 + ret)
                prices.append(price)
                count += 1
            df[i] = prices
        self.sim_price = df
        self.sim_price_last = df.iloc[-1]

    def predict_revenue_residual(self):
        drop_non_same_columns(self.count_revenue_monthly, self.rate_monthly)
        df = self.count_revenue_monthly.copy()
        df.drop(["total", "جمع"], axis=1, inplace=True)
        m_residual = 12 - self.last_m
        count_revenue_residual = pd.DataFrame(
            columns=self.products_lst, index=[self.future_year, self.future_year + 1]
        )
        rate_residual = pd.DataFrame(
            columns=self.products_lst, index=[self.future_year, self.future_year + 1]
        )
        c = 0

        for i in self.products_lst:
            ####### predict_count_revenue_residual ########

            try:
                model = pm.auto_arima(df[i], seasonal=True, m=12)
                vars()[f"model_count{c}"] = model
                self.__dict__[f"model_count{c}"] = vars()[f"model_count{c}"]
                res_prod = model.predict(m_residual)
                res_prod_next = model.predict(m_residual + 12).sum() - res_prod.sum()
                count_revenue_residual.loc[self.future_year, i] = res_prod.sum()
                count_revenue_residual.loc[self.future_year + 1, i] = res_prod_next
            except:
                count_revenue_residual.loc[self.future_year, i] = 0
                count_revenue_residual.loc[self.future_year + 1, i] = 0

            ###### Predict_rate_residual ############
            try:
                model = pm.auto_arima(self.rate_monthly[i], seasonal=True, m=12)
                vars()[f"model_rate{c}"] = model
                self.__dict__[f"model_rate{c}"] = vars()[f"model_rate{c}"]
                res_rate = model.predict(m_residual)
                res_rate = res_rate.mean()
                res_rate_next = model.predict(m_residual + 12)
                res_rate_next = res_rate_next[-12:].mean()
                rate_residual.loc[self.future_year, i] = res_rate
                rate_residual.loc[self.future_year + 1, i] = res_rate_next
            except:
                rate_residual.loc[self.future_year, i] = 0
                rate_residual.loc[self.future_year + 1, i] = 0
            c += 1
        count_revenue_residual = count_revenue_residual.applymap(
            lambda x: 0 if x < 0 else x
        )
        rate_residual = rate_residual.applymap(lambda x: 0 if x < 0 else x)
        count_revenue_residual["total"] = count_revenue_residual.sum(axis=1)
        self.count_revenue_residual = count_revenue_residual
        self.rate_residual = rate_residual

    def predict_salary(self):
        q_residual = 4 - self.last_q
        direct_salary = self.cost_quarterly[["دستمزد مستقیم تولید"]]
        model_direct_salary = pm.auto_arima(direct_salary, seasonal=True, m=4)
        self.model_direct_salary = model_direct_salary
        pred_direct_salary = pd.DataFrame(
            index=[self.future_year, self.future_year + 1], columns=["direct_salary"]
        )
        pred_direct_salary.loc[self.future_year] = (
            self.cost_done["دستمزد مستقیم تولید"].values[0]
            + model_direct_salary.predict(q_residual).sum()
        )
        pred_direct_salary.loc[self.future_year + 1] = (
            model_direct_salary.predict(4 + q_residual).sum()
            - model_direct_salary.predict(q_residual).sum()
        )
        self.pred_direct_salary = pred_direct_salary
        pred_overhead_salary = pd.DataFrame(
            index=[self.future_year, self.future_year + 1], columns=["overhead_salary"]
        )
        overhead_salary = self.overhead_quarterly[["هزینه حقوق و دستمزد"]]
        model_overhead_salary = pm.auto_arima(overhead_salary, seasonal=True, m=4)
        self.model_overhead_salary = model_overhead_salary
        pred_overhead_salary.loc[self.future_year] = (
            self.overhead_done["هزینه حقوق و دستمزد"].values[0]
            + model_overhead_salary.predict(q_residual).sum()
        )
        pred_overhead_salary.loc[self.future_year + 1] = (
            model_overhead_salary.predict(4 + q_residual).sum()
            - model_overhead_salary.predict(q_residual).sum()
        )
        self.pred_overhead_salary = pred_overhead_salary

    def predict_energy(self):
        q_residual = 4 - self.last_q
        energy = self.overhead_quarterly["هزینه انرژی (آب، برق، گاز و سوخت)"].copy()
        model = pm.auto_arima(energy, seasonal=True, m=4)
        self.model_energy = model
        pred_energy = pd.DataFrame(
            index=[self.future_year, self.future_year + 1], columns=["energy"]
        )
        pred_energy.loc[self.future_year] = (
            model.predict(q_residual).sum()
            + self.overhead_done["هزینه انرژی (آب، برق، گاز و سوخت)"].values[0]
        )
        pred_energy.loc[self.future_year + 1] = (
            model.predict(4 + q_residual).sum() - model.predict(q_residual).sum()
        )
        self.pred_energy = pred_energy

    def predict_transport(self):
        q_residual = 4 - self.last_q
        if "هزینه حمل و نقل و انتقال" in self.overhead_quarterly.columns:
            transport = self.overhead_quarterly["هزینه حمل و نقل و انتقال"].copy()
            model = pm.auto_arima(transport, seasonal=True, m=4)
            self.model_transport = model
            pred_transport = pd.DataFrame(
                index=[self.future_year, self.future_year + 1], columns=["transport"]
            )
            pred_transport.loc[self.future_year] = (
                self.overhead_done["هزینه حمل و نقل و انتقال"].values[0]
                + model.predict(q_residual).sum()
            )
            pred_transport.loc[self.future_year + 1] = (
                model.predict(4 + q_residual).sum() - model.predict(q_residual).sum()
            )
            self.pred_transport = pred_transport

    def predict_material_over(self):
        if "هزینه مواد مصرفی" in self.overhead_quarterly.columns:
            df = pd.DataFrame(columns=["direct_material", "over_material"])
            df["direct_material"] = self.cost_quarterly["مواد مستقیم مصرفی"]
            df["over_material"] = self.overhead_quarterly["هزینه مواد مصرفی"]
            df.dropna(inplace=True)
            x = df[["direct_material"]]
            y = df["over_material"]
            model = linear.LinearRegression()
            model.fit(x, y)
            self.model_material_over = model
            df["pred"] = model.predict(x)
            self.material_over_data = df
            pred_material_over = pd.DataFrame(
                columns=["material"], index=[self.future_year, self.future_year + 1]
            )
            pred_material_over["material"] = self.model_material_over.predict(
                self.pred_material
            )
            self.pred_material_over = pred_material_over

    def predict_other_operate(self):
        other_op = pd.DataFrame(columns=["other", "revenue"])
        other_op["revenue"] = self.income_quarterly["فروش"]
        other_op["other"] = self.income_quarterly[
            "خالص سایر درامدها (هزینه ها) ی عملیاتی"
        ]
        x = other_op[["revenue"]]
        y = other_op["other"]
        model = linear.LinearRegression()
        model.fit(x, y)
        pred_other_op = pd.DataFrame(
            columns=["other"], index=[self.future_year, self.future_year + 1]
        )
        pred_other_op.loc[self.future_year] = (
            model.predict(
                [[self.price_revenue_residual.loc[self.future_year, "total"]]]
            )
            + self.income_done["خالص سایر درامدها (هزینه ها) ی عملیاتی"].values[0]
        )
        pred_other_op.loc[self.future_year + 1] = model.predict(
            [[self.price_revenue_residual.loc[self.future_year + 1, "total"]]]
        )
        self.pred_other_op = pred_other_op

    def predict_other_non_operate(self):
        other_non_op = pd.DataFrame(columns=["other", "revenue"])
        other_non_op["revenue"] = self.income_quarterly["فروش"]
        other_non_op["other"] = self.income_quarterly[
            "خالص سایر درامدها و هزینه های غیرعملیاتی"
        ]
        x = other_non_op[["revenue"]]
        y = other_non_op["other"]
        model = linear.LinearRegression()
        model.fit(x, y)
        pred_other_non_op = pd.DataFrame(
            columns=["other"], index=[self.future_year, self.future_year + 1]
        )
        pred_other_non_op.loc[self.future_year] = (
            model.predict(
                [[self.price_revenue_residual.loc[self.future_year, "total"]]]
            )
            + self.income_done["خالص سایر درامدها و هزینه های غیرعملیاتی"].values[0]
        )
        pred_other_non_op.loc[self.future_year + 1] = model.predict(
            [[self.price_revenue_residual.loc[self.future_year + 1, "total"]]]
        )
        self.pred_other_non_op = pred_other_non_op

    def predict_tax(self):
        tax_data = pd.DataFrame(columns=["tax", "pre_tax"])
        tax_data["tax"] = -self.income_yearly["مالیات"]
        tax_data["pre_tax"] = self.income_yearly[
            "سود (زیان) خالص عملیات در حال تداوم قبل از مالیات"
        ]
        y = tax_data["tax"]
        x = tax_data[["pre_tax"]]
        model = linear.LinearRegression()
        model.fit(x, y)
        tax_data["pred"] = model.predict(x)
        self.tax_data = tax_data
        self.model_tax = model
        pred_tax = model.predict(
            self.pred_income[
                ["سود (زیان) خالص عملیات در حال تداوم قبل از مالیات"]
            ].iloc[-2:]
        )
        self.pred_income.loc[self.future_year, "مالیات"] = -pred_tax[0]
        self.pred_income.loc[self.future_year + 1, "مالیات"] = -pred_tax[1]
        self.pred_income.update_dependent_columns()


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
        # w, best_loss = ps.pso(self.cost_function, LB, UB, swarmsize=40, maxiter=60)
        # return w, best_loss

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
        plt.plot(self.total_industry["سود (زیان) خالص"], marker="o")
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
        plt.plot(self.total_industry_quarterly_12["سود (زیان) خالص"], marker="o")
        plt.title("Total_Net_quarterly")


class Time_Series:
    def __init__(self, df, model, type_step):
        self.model = model
        self.type_step = type_step
        self.df = df
        self.series = df.to_numpy()
        if type_step == "multi":
            self.Tx = int(input("Enter Tx:"))
            self.Ty = int(input("Enter Ty:"))
        if type_step == "single":
            self.ntest = int(input("Enter ntest:"))
            self.T = int(input("Enter of desired lags :"))
        self.create_x_y()
        self.fit_model()
        self.create_resault()

    def create_x_y(self):
        if self.type_step == "multi":
            X = []
            Y = []
            for t in range(len(self.series) - self.Tx - self.Ty + 1):
                x = self.series[t : t + self.Tx]
                y = self.series[t + self.Tx : t + self.Tx + self.Ty]
                X.append(x)
                Y.append(y)
            X = np.array(X).reshape(-1, self.Tx)
            Y = np.array(Y).reshape(-1, self.Ty)
            self.X = X
            self.Y = Y
            Xtrain = X[:-1]
            Ytrain = Y[:-1]
            Xtest = X[-1:]
            Ytest = Y[-1:]
            self.Xtrain = Xtrain
            self.Ytrain = Ytrain
            self.Xtest = Xtest
            self.Ytest = Ytest
            train_idx = self.df.index[self.Tx : len(self.series) - self.Ty]
            test_idx = self.df.index[-self.Ty :]
            self.train_idx = train_idx
            self.test_idx = test_idx
        if self.type_step == "single":
            X = []
            Y = []
            for t in range(len(self.series) - self.T):
                x = self.series[t : t + self.T]
                y = self.series[t + self.T]
                X.append(x)
                Y.append(y)
            X = np.array(X).reshape(-1, self.T)
            Y = np.array(Y)
            self.X = X
            self.Y = Y
            Xtrain = X[: -self.ntest]
            Ytrain = Y[: -self.ntest]
            Xtest = X[-self.ntest :]
            Ytest = Y[-self.ntest :]
            self.Xtrain = Xtrain
            self.Ytrain = Ytrain
            self.Xtest = Xtest
            self.Ytest = Ytest
            train_idx = self.df.index[self.T : len(self.series) - self.ntest]
            test_idx = self.df.index[-self.ntest :]
            self.train_idx = train_idx
            self.test_idx = test_idx

    def fit_model(self):
        self.model.fit(self.Xtrain, self.Ytrain)
        if self.type_step == "single":
            self.pred = self.predict_future(self.Xtest[0], self.ntest)

        if self.type_step == "multi":
            self.pred = self.model.predict(self.Xtest).flatten()

    def predict_future(self, last_x, steps):
        if self.type_step == "single":
            pred = []
            while len(pred) < steps:
                p = self.model.predict(last_x.reshape(1, -1))[0]
                pred.append(p)
                last_x = np.roll(last_x, -1)
                last_x[-1] = p
        if self.type_step == "multi":
            pred = self.model.predict(last_x.reshape(1, -1)).flatten()
        return pred

    def create_resault(self):
        res = self.df.loc[self.test_idx]
        res.loc[self.test_idx, "pred"] = self.pred
        self.res = res

    def plot_resault(self):
        self.res.plot()
        r2 = r2_score(self.Ytest.flatten(), self.pred)
        print("R2:", r2)


class TimeSeries_diff:
    def __init__(self, df, model, type_step):
        self.model = model
        self.type_step = type_step
        df.rename(columns={df.columns[0]: "Data"}, inplace=True)
        df["diff"] = df.diff(1)
        df.dropna(inplace=True)
        self.df = df
        self.series = df["diff"].to_numpy()
        if type_step == "multi":
            self.Tx = int(input("Enter Tx:"))
            self.Ty = int(input("Enter Ty:"))
        if type_step == "single":
            self.ntest = int(input("Enter ntest:"))
            self.T = int(input("Enter of desired lags :"))
        self.create_x_y()
        self.fit_model()
        self.create_resault()

    def create_x_y(self):
        if self.type_step == "multi":
            X = []
            Y = []
            for t in range(len(self.series) - self.Tx - self.Ty + 1):
                x = self.series[t : t + self.Tx]
                y = self.series[t + self.Tx : t + self.Tx + self.Ty]
                X.append(x)
                Y.append(y)
            X = np.array(X).reshape(-1, self.Tx)
            Y = np.array(Y).reshape(-1, self.Ty)
            self.X = X
            self.Y = Y
            Xtrain = X[:-1]
            Ytrain = Y[:-1]
            Xtest = X[-1:]
            Ytest = Y[-1:]
            self.Xtrain = Xtrain
            self.Ytrain = Ytrain
            self.Xtest = Xtest
            self.Ytest = Ytest
            train_idx = self.df.index[self.Tx : len(self.series) - self.Ty]
            test_idx = self.df.index[-self.Ty :]
            self.train_idx = train_idx
            self.test_idx = test_idx
            self.last_train = self.df.iloc[-self.Ty - 1]["Data"]
        if self.type_step == "single":
            X = []
            Y = []
            for t in range(len(self.series) - self.T):
                x = self.series[t : t + self.T]
                y = self.series[t + self.T]
                X.append(x)
                Y.append(y)
            X = np.array(X).reshape(-1, self.T)
            Y = np.array(Y)
            self.X = X
            self.Y = Y
            Xtrain = X[: -self.ntest]
            Ytrain = Y[: -self.ntest]
            Xtest = X[-self.ntest :]
            Ytest = Y[-self.ntest :]
            self.Xtrain = Xtrain
            self.Ytrain = Ytrain
            self.Xtest = Xtest
            self.Ytest = Ytest
            train_idx = self.df.index[self.T : len(self.series) - self.ntest]
            test_idx = self.df.index[-self.ntest :]
            self.train_idx = train_idx
            self.test_idx = test_idx
            self.last_train = self.df.iloc[-self.ntest - 1]["Data"]

    def fit_model(self):
        self.model.fit(self.Xtrain, self.Ytrain)
        if self.type_step == "single":
            self.pred = self.predict_future(self.Xtest[0], self.ntest)[0]
            self.pred_y = self.predict_future(self.Xtest[0], self.ntest)[1]

        if self.type_step == "multi":
            self.pred = self.model.predict(self.Xtest)[0].flatten()
            self.pred_y = self.predict_future(self.Xtest[0], self.Tx)[1].flatten()

    def predict_future(self, last_x, steps):
        if self.type_step == "single":
            pred = []
            pred_y = []
            last = self.last_train
            while len(pred) < steps:
                p = self.model.predict(last_x.reshape(1, -1))[0]
                pred.append(p)
                last_x = np.roll(last_x, -1)
                last_x[-1] = p
                pred_y.append(last + p)
                last = pred_y[-1]
                self.pred_y = pred_y
        if self.type_step == "multi":
            pred = self.model.predict(last_x.reshape(1, -1)).flatten()
            pred_y = self.last_train + np.cumsum(pred)
        return pred, pred_y

    def create_resault(self):
        res = self.df.loc[self.test_idx]
        res.loc[self.test_idx, "pred_diff"] = self.pred
        res.loc[self.test_idx, "pred_y"] = self.pred_y
        self.res = res

    def plot_resault(self):
        self.res[["Data", "pred_y"]].plot()
        r2 = r2_score(self.Ytest.flatten(), self.pred)
        print("R2:", r2)


class TimeSeries_direct:
    def __init__(self, model, df, T, ntest):
        self.model = model
        self.T = T
        self.ntest = ntest
        df.rename(columns={df.columns[0]: "Data"}, inplace=True)
        df["diff"] = df.diff(1)
        df.dropna(inplace=True)
        zero = df[df["diff"] == 0]
        df.drop(zero.index, inplace=True)
        self.df = df

        self.series = df["diff"].to_numpy()
        self.target = (self.series > 0) * 1
        self.create_x_y()
        self.fit_model()

    def create_x_y(self):
        X = []
        Y = []
        for t in range(len(self.series) - self.T):
            x = self.series[t : t + self.T]
            X.append(x)
            y = self.target[t + self.T]
            Y.append(y)
        X = np.array(X).reshape(-1, self.T)
        Y = np.array(Y)
        self.X = X
        self.Y = Y
        Xtrain = X[: -self.ntest]
        Ytrain = Y[: -self.ntest]
        Xtest = X[-self.ntest :]
        Ytest = Y[-self.ntest :]
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest

    def fit_model(self):
        self.model.fit(self.Xtrain, self.Ytrain)
        ypred = self.model.predict(self.Xtest)
        self.ypred = ypred

    def resault(self):
        score = accuracy_score(self.ypred, self.Ytest)
        print("acuracy_score:", score)
