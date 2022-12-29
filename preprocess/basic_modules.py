import os
import sys
import shutil
import re
import platform
import pandas as pd
import win32com.client as win32

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import tee
from statics.setting import *


def benfords_law(nums):
    ones = list(filter(lambda x: True if str(x)[0] == "1" else False, nums))
    return len(ones) / len(nums)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def all_dict_values(data: dict):
    for v in data.values():
        if isinstance(v, dict):
            yield from all_dict_values(v)
        else:
            yield v


def only_zero_inequality(n):
    """just show nums != 0"""
    try:
        n = float(n)

    except:
        if type(n) != float:
            return None

    if n == 0:
        return None

    else:
        return n


def clarify_number(a, seprator=",", n=2):
    """
    12345678 => 12,345,678

    -12345678 => (12,345,678)

    12345678.1234 => 12,345,678.12
    """

    is_float = False
    is_negative = False

    try:
        a = float(a)

        if int(a) != a:
            is_float = True
            float_part = str(round(a, n)).split(".")[1]

        if a < 0:
            is_negative = True
            a = abs(a)

        a = str(int(a))

    except:
        return a

    l = len(a) % 3
    b = ""
    b += a[:l]
    for i in range(len(a) // 3):
        b += seprator
        b += a[l : l + 3]
        l += 3

    if b[0] == seprator:
        b = b[1:]

    if is_float:
        if n != 0:
            b = b + "." + float_part

    if is_negative:
        b = "(" + b + ")"

    return b


def to_digits(a):
    """
    (۲۵۴,۱۵۹) : -254159
    ۴۳۹,۶۲۸ : 439628
    """

    if isinstance(a, float) or isinstance(a, int):
        return a

    if isinstance(a, str):
        en_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        fa_digits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

        b = ""
        for s in a:
            if s in "".join(en_digits) + "".join(fa_digits):
                for i in range(0, 10):
                    if s == fa_digits[i]:
                        s = en_digits[i]
                b = b + s

        is_negative = False
        if "(" in a and ")" in a:
            is_negative = True

        if b == "":
            return False

        elif not is_negative:
            b = int(b)

        elif is_negative:
            b = int(b)

        return b

    else:
        return False


def best_table_id(data_tables):
    """get a data tables and return useful data table id"""

    tester = 0
    for i in range(0, len(data_tables)):
        if len(data_tables[i]) > tester:
            tester = len(data_tables[i])
            table_id = i

    return table_id


def to_useful_excel(file_path):
    """convert bourseview excel to new excel that pandas use it"""
    if platform.system() == "Windows":
        try:
            excel = win32.gencache.EnsureDispatch("Excel.Application")
        except AttributeError:
            # remove cache and try again
            MODULE_LIST = [m.__name__ for m in sys.modules.values()]
            for module in MODULE_LIST:
                if re.match(r"win32com\.gen_py\..+", module):
                    del sys.modules[module]
            shutil.rmtree(
                os.path.join(os.environ.get("LOCALAPPDATA"), "Temp", "gen_py")
            )
            excel = win32.gencache.EnsureDispatch("Excel.Application")

        # open and save excel file
        excel.Workbooks.Open(file_path).Save()
        excel.Application.Quit()


def move_last_file(new_path):
    """move last DB file to new_path"""
    # find latest downloaded file
    filename = max(
        [f for f in os.listdir(DB)],
        key=lambda x: os.path.getctime(os.path.join(DB, x)),
    )

    # replace last file
    if ".xlsx" in filename:
        os.replace(
            f"{DB}/{filename}",
            new_path,
        )


def list_stock_files(stock_name):
    """return all folders and files of stock in database"""
    stock_dirs = []
    stock_files = []
    for path, subdirs, files in os.walk(
        f"{INDUSTRIES_PATH}/{watchlist[stock_name]['indus']}/{stock_name}"
    ):
        for dir in subdirs:
            stock_dirs.append(os.path.join(path, dir).replace("\\", "/"))

        for name in files:
            stock_files.append(os.path.join(path, name).replace("\\", "/"))

    return stock_dirs, stock_files


def get_excel_nums(file_path):
    df = pd.read_excel(file_path)
    nums = []
    for i in df.items():
        for j in df[i[0]].items():
            num = j[1]
            if pd.notna(num) and to_digits(num):
                nums.append(abs(to_digits(num)))

    return nums


def create_database_structure():
    """create database folders based on watchlist"""
    for stock, info in watchlist.items():
        for file in all_dict_values(structure):
            s = "/".join(file.split("/")[:-1])

            Path(f"{INDUSTRIES_PATH}/{info['indus']}/{stock}/{s}").mkdir(
                parents=True, exist_ok=True
            )

    Path(MACRO_PATH).mkdir(parents=True, exist_ok=True)
    Path(FOREX_PATH).mkdir(parents=True, exist_ok=True)
    Path(PICKLES_PATH).mkdir(parents=True, exist_ok=True)


def find_deficiencies(stock_name):

    base_files = [
        f"{INDUSTRIES_PATH}/{watchlist[stock_name]['indus']}/{stock_name}/{s}"
        for s in all_dict_values(structure)
        if ".xlsx" in s
    ]

    files = [file for file in base_files if not Path(file).exists()]
    deficiencies = defaultdict(list)
    stock_types, time_types = [], []
    pe = False
    opt = False
    eps = False

    for file in files:
        if "eps.xlsx" in file:
            eps = True
        elif "opt.xlsx" in file:
            opt = True
        elif "pe.xlsx" in file:
            pe = True

        else:
            stock_types.append(file.split("/")[6].split(".")[0])
            time_types.append(file.split("/")[7].split(".")[0].split("_")[0])

    for key, value in zip(stock_types, time_types):
        deficiencies[key].append(value)
    deficiencies = dict(deficiencies)

    for i in deficiencies:
        deficiencies[i] = list(set(deficiencies[i]))

    return deficiencies, pe, opt, eps


def check_stock_files(stock_name, user_year=0, user_month=0, user_quarter=0):
    base_files = [
        f"{INDUSTRIES_PATH}/{watchlist[stock_name]['indus']}/{stock_name}/{s}"
        for s in all_dict_values(structure)
        if ".xlsx" in s
    ]

    for file in base_files:
        if Path(file).exists():
            file_size = (os.stat(file).st_size) / (1024 * 1024)
            file_ctime = JalaliDate(datetime.fromtimestamp(os.path.getctime(file)))

            if file_size > 1:
                print("large file :", file)

            try:
                df = pd.read_excel(file)
                if df.applymap(only_zero_inequality).isnull().all().all():
                    print("empty file :", file)

            except:
                print("old format : ", file)

            try:
                stock_type = file.split("/")[6]
                stock_time = file.split("/")[7].split(".")[0].split("_")[0]

            except:
                pass

            try:
                excel_timeids = re.findall(regex_en_timeid_q, str(df.loc[6]))
                excel_years = list(map(lambda x: int(x.split("/")[0]), excel_timeids))
                excel_months = list(map(lambda x: int(x.split("/")[1]), excel_timeids))
                excel_steps = [abs(a - b) for a, b in pairwise(excel_months)]
                excel_step = max(excel_steps, key=excel_steps.count)

                steptypes = {1: "monthly", 3: "quarterly", 0: "yearly"}
                if steptypes[excel_step] != stock_time:
                    print("unmatch time :", file)

                if stock_time == "monthly" and user_month > excel_months[-1]:
                    print("old data :", file, file_ctime)

                if stock_time == "quarterly" and user_quarter > excel_months[-1]:
                    print("old data :", file, file_ctime)

                if stock_time == "yearly" and user_year > excel_years[-1]:
                    print("old data :", file, file_ctime)

            except:
                pass

            try:
                excel_author = df["Unnamed: 1"][0]
                excel_type = df["Unnamed: 1"][4]
                excel_token = (df["Unnamed: 1"][3]).replace("\u200c", "").split("-")[0]

                stock_types = {
                    "balancesheet": "Balance Sheet",
                    "income": "Income Statements",
                    "cashflow": "Cash Flow",
                    "product": "تولید و فروش",
                    "cost": "بهای تمام شده",
                    "official": "هزینه های عمومی و اداری",
                    "pe": "تاریخچه قیمت",
                }

                if excel_author != "Pouya Finance":
                    print("not bourseview : ", file)

                if excel_type != stock_types[stock_type]:
                    print("unmatch type : ", file)

                if excel_token != watchlist[stock_name]["token"]:
                    print("unmatch name :", file)

            except:
                pass
