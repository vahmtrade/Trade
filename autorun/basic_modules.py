import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from persiantools.jdatetime import JalaliDate

from statics.setting import DB, watchlist


base_folders = [
    "balancesheet",
    "income",
    "cashflow",
    "product",
    "cost",
    "official",
    "pe",
    "analyse",
    "detail_trade",
]


def to_digits(string):
    """get a string like '(۲۵۴,۱۵۹,۳۴۷)' [negative] or '۴۳۹,۶۲۸,۱۹۸' [positive] and return a number"""

    if isinstance(string, str) == False:
        return 1

    en_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fa_digits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

    number = ""
    flag = 0

    if "(" and ")" in string:
        flag = 1

    for s in string:
        if s in "0123456789۰۱۲۳۴۵۶۷۸۹":
            for i in range(0, 10):
                if s == fa_digits[i]:
                    s = en_digits[i]
            number = number + s

    try:
        if flag == 0:
            number = int(number)
        if flag == 1:
            number = -int(number)

    except:
        number = number

    return number


def best_table_id(data_tables):
    """get a data tables and return useful data table id"""

    tester = 0
    for i in range(0, len(data_tables)):
        if len(data_tables[i]) > tester:
            tester = len(data_tables[i])
            table_id = i

    return table_id


def create_database_structure():
    """create database folders based on watchlist"""

    for stock, info in watchlist.items():

        for i in base_folders:

            if i == "income":
                for j in ("yearly", "quarterly"):
                    Path(f"{DB}/industries/{info['indus']}/{stock}/{i}/{j}/").mkdir(
                        parents=True, exist_ok=True
                    )

            else:
                Path(f"{DB}/industries/{info['indus']}/{stock}/{i}/").mkdir(
                    parents=True, exist_ok=True
                )


def list_stock_files(stock_name):
    """return all folders and files of stock in database"""
    stock_dirs = []
    stock_files = []
    for path, subdirs, files in os.walk(
        f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}"
    ):
        for dir in subdirs:
            stock_dirs.append(os.path.join(path, dir).replace("\\", "/"))

        for name in files:
            stock_files.append(os.path.join(path, name).replace("\\", "/"))

    return stock_dirs, stock_files


def update_stock_files(stock_name):
    """delete unnecessary and add deficiency data"""

    # files that must every stock have it
    stock_folder = f"{DB}/industries/{watchlist[stock_name]['indus']}"
    base_files = [
        f"{stock_folder}/{stock_name}/balancesheet/quarterly.xlsx",
        f"{stock_folder}/{stock_name}/balancesheet/yearly.xlsx",
        f"{stock_folder}/{stock_name}/cashflow/quarterly.xlsx",
        f"{stock_folder}/{stock_name}/cashflow/yearly.xlsx",
        f"{stock_folder}/{stock_name}/cost/quarterly.xlsx",
        f"{stock_folder}/{stock_name}/cost/yearly.xlsx",
        f"{stock_folder}/{stock_name}/income/quarterly/dollar.xlsx",
        f"{stock_folder}/{stock_name}/income/quarterly/rial.xlsx",
        f"{stock_folder}/{stock_name}/income/yearly/dollar.xlsx",
        f"{stock_folder}/{stock_name}/income/yearly/rial.xlsx",
        f"{stock_folder}/{stock_name}/official/quarterly.xlsx",
        f"{stock_folder}/{stock_name}/official/yearly.xlsx",
        f"{stock_folder}/{stock_name}/pe/pe.xlsx",
        f"{stock_folder}/{stock_name}/pe/forward.xlsx",
        f"{stock_folder}/{stock_name}/product/monthly.xlsx",
        f"{stock_folder}/{stock_name}/product/monthly_seprated.xlsx",
        f"{stock_folder}/{stock_name}/product/quarterly.xlsx",
        f"{stock_folder}/{stock_name}/product/quarterly_seprated.xlsx",
        f"{stock_folder}/{stock_name}/product/yearly.xlsx",
        f"{stock_folder}/{stock_name}/product/yearly_seprated.xlsx",
        f"{stock_folder}/{stock_name}/eps.xlsx",
        f"{stock_folder}/{stock_name}/opt.xlsx",
    ]

    all_files = list_stock_files(stock_name)

    # delete unnecessary files
    for file in all_files[1]:
        if "/detail_trade" not in file and "/analyse" not in file:
            if file not in base_files:
                print("unnecessary file : ", file)
                os.remove(file)

    # delete empty folders
    for dir in all_files[0]:
        if "/detail_trade" not in dir and "/analyse" not in dir:
            if len(os.listdir(dir)) == 0:
                print("unnecessary folder : ", dir)
                os.rmdir(dir)

    # search in needed files
    for b in base_files:
        if Path(b).exists() == False:
            # show deficiency files
            print("deficiency : ", b)

        else:
            # get creation time of file
            t = JalaliDate(datetime.fromtimestamp(os.path.getctime(b)))

            # check sanity of bourseview excels
            try:
                if (
                    "eps.xlsx" not in b
                    and "opt.xlsx" not in b
                    and "forward.xlsx" not in b
                ):
                    sample = pd.read_excel(b)["Unnamed: 1"][0]
                    if sample != "Pouya Finance":
                        print(b, sample)

            except:
                print(b)
