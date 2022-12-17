import os
import sys
import shutil
import re
import platform
import win32com.client as win32

from pathlib import Path

from statics.setting import *


def all_dict_values(d: dict):
    for v in d.values():
        if isinstance(v, dict):
            yield from all_dict_values(v)
        else:
            yield v


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


def create_database_structure():
    """create database folders based on watchlist"""
    for stock, info in watchlist.items():
        for file in all_dict_values(structure):
            s = ""
            for i in file.split("/")[:-1]:
                s += "/" + i

            Path(f"{INDUSTRIES_PATH}/{info['indus']}/{stock}{s}").mkdir(
                parents=True, exist_ok=True
            )


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
            shutil.rmtree(os.path.join(os.environ.get("LOCALAPPDATA"), "Temp", "gen_py"))
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


def clarify_number(a, seprator=",", n=2):
    """12345678 => 12,345,678

    -12345678 => (12,345,678)

    12345678.1234 => 12,345,678.12"""

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


def to_digits(string):
    """get a string like '(۲۵۴,۱۵۹,۳۴۷)' [negative] or '۴۳۹,۶۲۸,۱۹۸' [positive] and return a number"""

    if isinstance(string, str) == False:
        return 1

    en_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fa_digits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

    number = ""
    flag = 0

    if "(" in string and ")" in string:
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
