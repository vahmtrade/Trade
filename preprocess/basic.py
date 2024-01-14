import os
import sys
import shutil
import subprocess
import re
import platform
import jdatetime
import pandas as pd

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import tee

from statics.setting import *

if platform.system() == "Windows":
    import win32com.client as win32  # type: ignore


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def all_dict_values(data: dict):
    """data : {"A": 1,"B": {"C": 2,"D": {"E": 3}}} => [1,2,3]"""
    for v in data.values():
        if isinstance(v, dict):
            yield from all_dict_values(v)
        else:
            yield v


def only_zero_inequality(n):
    """Accepts only non-zero values"""
    if isinstance(n, (int, float)) and n != 0:
        return n
    else:
        return None


def remove_non_digit(content):
    if isinstance(content, (float, int)):
        return content
    en_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fa_digits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]
    digits = "".join(
        [
            en_digits[fa_digits.index(d)] if d in fa_digits else d
            for d in str(content)
            if d.isdigit()
        ]
    )

    return digits


def to_digits(content):
    """
    (۲۵۴,۱۵۹) => -254159

    ۴۳۹,۶۲۸ => 439628
    """
    digits = remove_non_digit(content)
    return (
        int(digits) * (-1 if "(" in str(content) and ")" in str(content) else 1)
        if digits
        else False
    )


def convert_timeid(t, option="gregorian"):
    """
    t : 14010701

    option : gregorian,persian
    """
    t = remove_non_digit(t)
    year, month, day = int(t[:4]), int(t[4:6]), int(t[6:8])
    hijri_date = jdatetime.date(year, month, day)

    persian_date = f'{hijri_date.jweekday()} {hijri_date.strftime("%d %B %Y")}'
    gregorian_date = (hijri_date.togregorian()).isoformat()

    data = {"gregorian": gregorian_date, "persian": persian_date}
    return data[option]


def benfords_law(nums):
    ones = list(filter(lambda x: True if str(x)[0] == "1" else False, nums))
    return len(ones) / len(nums)


def clarify_number(a, seprator=",", n=2):
    """-1234.56789 => (1,234.56)"""

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
    for _ in range(len(a) // 3):
        b += seprator
        b += a[l : l + 3]
        l += 3

    if b[0] == seprator:
        b = b[1:]

    if is_float:
        if n != 0:
            b = f"{b}.{float_part}"

    if is_negative:
        b = f"({b})"

    return b


def move_last_file(new_path, base_path=DB):
    filename = max(
        os.listdir(DB),
        key=lambda x: os.path.getctime(os.path.join(DB, x)),
    )
    old_path = f"{base_path}/{filename}"
    if os.path.isfile(old_path):
        os.replace(old_path, new_path)


def load_windows_app(app_name):
    try:
        app = win32.gencache.EnsureDispatch(app_name)
    except AttributeError:
        # remove cache and try again
        MODULE_LIST = [m.__name__ for m in sys.modules.values()]
        for module in MODULE_LIST:
            if re.match(r"win32com\.gen_py\..+", module):
                del sys.modules[module]
        shutil.rmtree(os.path.join(os.environ.get("LOCALAPPDATA"), "Temp", "gen_py"))
        app = win32.gencache.EnsureDispatch(app_name)

    return app


def resave_excel(file_path):
    folder, filename = os.path.split(file_path)

    if platform.system() == "Windows":
        excel = load_windows_app("Excel.Application")
        workbook = excel.Workbooks.Open(file_path)
        workbook.Save()
        excel.Application.Quit()

    if platform.system() == "Linux":
        subprocess.run(["libreoffice", "--convert-to", "xlsx", "--headless", file_path])
        os.replace(f"./{filename}", file_path)


def save_as_file(file_path, ext):
    folder, filename = os.path.split(file_path)

    if platform.system() == "Windows":
        exts = {"xlsx": 51, "xls": 56, "html": 44}
        new_name = os.path.splitext(file_path)[0] + "." + ext
        excel = load_windows_app("Excel.Application")
        workbook = excel.Workbooks.Open(file_path)
        workbook.SaveAs(new_name.replace("/", "\\"), FileFormat=exts[ext])
        excel.Application.Quit()
        os.remove(file_path)

    if platform.system() == "Linux":
        new_name = os.path.splitext(filename)[0] + "." + ext
        subprocess.run(["libreoffice", "--convert-to", ext, "--headless", file_path])
        os.remove(file_path)
        shutil.move(f"./{new_name}", f"{folder}/{new_name}")


def get_excel_nums(file_path):
    df = pd.read_excel(file_path)
    nums = []
    for i in df.items():
        for j in df[i[0]].items():
            num = j[1]
            if pd.notna(num) and to_digits(num):
                nums.append(abs(to_digits(num)))

    return nums


def essential_stock_files(stock_name):
    return [
        f"{INDUSPATH}/{wl_prod[stock_name]['indus']}/{stock_name}/{i}"
        for i in all_dict_values(structure)
        if ".xlsx" in i
    ]


def list_stock_files(stock_name):
    stock_dirs = []
    stock_files = []
    for path, subdirs, files in os.walk(
        f"{INDUSPATH}/{wl_prod[stock_name]['indus']}/{stock_name}"
    ):
        for dir in subdirs:
            stock_dirs.append(os.path.join(path, dir).replace("\\", "/"))

        for name in files:
            stock_files.append(os.path.join(path, name).replace("\\", "/"))

    return stock_dirs, stock_files


def create_database_structure():
    for stock, info in wl_prod.items():
        for file in all_dict_values(structure):
            s = "/".join(file.split("/")[:-1])

            Path(f"{INDUSPATH}/{info['indus']}/{stock}/{s}").mkdir(
                parents=True, exist_ok=True
            )

    Path(MACROPATH).mkdir(parents=True, exist_ok=True)
    Path(FOREXPATH).mkdir(parents=True, exist_ok=True)
    Path(PKLPATH).mkdir(parents=True, exist_ok=True)


def filepath_info(filepath):
    article_type, per_article_type, time_type = "", "", ""
    try:
        for i in article_types_dict:
            if i in filepath:
                article_type = i

        per_article_type = article_types_dict[article_type]

    except Exception as err:
        print("cant get type:", filepath, err)

    try:
        for i in ["monthly", "quarterly", "yearly"]:
            if i in filepath:
                time_type = i

    except Exception as err:
        print("cant get time:", filepath, err)

    return [article_type, per_article_type, time_type]


def find_deficiencies(stock_name):
    """deficiencies, pe, opt, eps"""

    base_files = essential_stock_files(stock_name)
    missing_files = [i for i in base_files if not Path(i).exists()]

    article_types, time_types = [], []
    pe, opt, eps = False, False, False

    for file in missing_files:
        if "eps.xlsx" in file:
            eps = True
        elif "opt.xlsx" in file:
            opt = True
        elif "pe.xlsx" in file:
            pe = True

        else:
            article_type, per_article_type, time_type = filepath_info(file)
            article_types.append(article_type)
            time_types.append(time_type)

    deficiencies = defaultdict(list)

    for key, value in zip(article_types, time_types):
        deficiencies[key].append(value)
    deficiencies = dict(deficiencies)

    for i in deficiencies:
        deficiencies[i] = list(set(deficiencies[i]))

    return deficiencies, pe, opt, eps


def check_stock_files(
    stock_name, last_year=False, last_quarter=False, last_month=False, delete=False
):
    failed = []
    base_files = essential_stock_files(stock_name)

    for file in base_files:
        if Path(file).exists():
            file_size = (os.stat(file).st_size) / (1024 * 1024)
            file_ctime = JalaliDate(datetime.fromtimestamp(os.path.getctime(file)))

            if file_size > 1:
                print("large file :", file)
                failed.append(file)

            try:
                df = pd.read_excel(file)
                if df.applymap(only_zero_inequality).isnull().all().all():
                    print("empty file :", file)
                    failed.append(file)

            except Exception as err:
                print("old format : ", file, err)
                failed.append(file)

            try:
                article_type, per_article_type, time_type = filepath_info(file)

            except Exception as err:
                print("cant get filepath info :", file, err)

            try:
                excel_timeids = re.findall(regex_en_timeid_q, str(df.loc[6]))
                excel_years = list(map(lambda x: int(x.split("/")[0]), excel_timeids))
                excel_months = list(map(lambda x: int(x.split("/")[1]), excel_timeids))
                excel_steps = [abs(a - b) for a, b in pairwise(excel_months)]
                excel_step = max(excel_steps, key=excel_steps.count)
                excel_last_quarter = ((excel_months[-1] - 1) // 3) + 1

                steptypes = {1: "monthly", 3: "quarterly", 0: "yearly"}
                if steptypes[excel_step] != time_type:
                    print("unmatch time :", file)
                    failed.append(file)

                if time_type == "monthly" and last_month != False:
                    if str(last_month) != str(excel_months[-1]):
                        print("old data monthly :", file, excel_months[-1])
                        failed.append(file)

                if time_type == "quarterly" and last_quarter != False:
                    if str(last_quarter) != str(excel_last_quarter):
                        print("old data quarterly :", file, excel_months[-1])
                        failed.append(file)

                if time_type == "yearly" and last_year != False:
                    if str(last_year) != str(excel_years[-1]):
                        print("old data yearly :", file, excel_years[-1])
                        failed.append(file)

            except Exception as err:
                # TODO : remove pe,opt,eps from this exception
                # print("cant get excel timeids :", file, err)
                print(err)

            try:
                excel_author = df["Unnamed: 1"][0]
                excel_type = df["Unnamed: 1"][4]
                excel_token = (df["Unnamed: 1"][3]).replace("\u200c", "").split("-")[0]

                if excel_author != "Pouya Finance":
                    print("not bourseview : ", file)
                    failed.append(file)

                if excel_type != per_article_type:
                    print("unmatch type : ", file, per_article_type)
                    failed.append(file)

                if excel_token != wl_prod[stock_name]["token"]:
                    print("unmatch name :", file)
                    failed.append(file)

            except Exception as err:
                # TODO : remove pe,opt,eps from this exception
                # print("cant get excel author :", file, err)
                print(err)

    failed = list(set(failed))
    if delete:
        for i in failed:
            os.remove(i)
