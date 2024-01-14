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
        os.listdir(base_path),
        key=lambda x: os.path.getctime(os.path.join(base_path, x)),
    )
    old_path = f"{base_path}/{filename}"
    if os.path.isfile(old_path):
        os.replace(old_path, new_path)


def load_windows_app(app_name):
    try:
        app = win32.gencache.EnsureDispatch(app_name)
    except AttributeError:
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


def all_dict_values(data: dict):
    """data : {"A": 1,"B": {"C": 2,"D": {"E": 3}}} => [1,2,3]"""
    for v in data.values():
        if isinstance(v, dict):
            yield from all_dict_values(v)
        else:
            yield v


def essential_stock_files(stock_name):
    return [
        f"{INDUSPATH}/{wl_prod[stock_name]['indus']}/{stock_name}/{i}"
        for i in all_dict_values(structure)
        if ".xlsx" in i
    ]


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


def find_deficiencies(stock_name):
    base_files = essential_stock_files(stock_name)
    missing_filepaths = [i for i in base_files if not Path(i).exists()]
    article_types, time_types, pe, opt, eps = [], [], False, False, False

    for filepath in missing_filepaths:
        if "eps.xlsx" in filepath:
            eps = True
        elif "opt.xlsx" in filepath:
            opt = True
        elif "pe.xlsx" in filepath:
            pe = True
        else:
            article_type = next((i for i in article_types_dict if i in filepath), "")
            time_type = next(
                (i for i in ["monthly", "quarterly", "yearly"] if i in filepath), ""
            )
            article_types.append(article_type)
            time_types.append(time_type)

    deficiencies = defaultdict(list)
    for key, value in zip(article_types, time_types):
        deficiencies[key].append(value)

    return {i: list(set(deficiencies[i])) for i in deficiencies}, pe, opt, eps


def not_pe_opt_eps(filepath):
    return not any(x in filepath for x in ["pe.xlsx", "eps.xlsx", "opt.xlsx"])


def filepath_info(filepath):
    if not_pe_opt_eps(filepath):
        article_type = next((i for i in article_types_dict if i in filepath), "")
        per_article_type = article_types_dict.get(article_type, "")
        time_types = ["monthly", "quarterly", "yearly"]
        time_type = next((i for i in time_types if i in filepath), "")
        return [article_type, per_article_type, time_type]

    else:
        return [False, False, False]


def is_stock_file_large_size(filepath):
    """Returns True if the file size is larger than 1 MB, False otherwise"""

    file_size = (os.stat(filepath).st_size) / (1024 * 1024)
    file_ctime = JalaliDate(datetime.fromtimestamp(os.path.getctime(filepath)))
    if file_size > 1:
        print("large file :", filepath, file_ctime)
        return True
    else:
        return False


def is_stock_file_empty(df, filepath):
    """Returns True if the file is empty, False otherwise"""

    if df.applymap(only_zero_inequality).isnull().all().all():
        print("empty file :", filepath)
        return True
    else:
        return False


def is_stock_file_not_match_time(
    df, filepath, time_type, last_year, last_quarter, last_month
):
    """Returns True if the file time does not match the expected time, False otherwise"""

    if not_pe_opt_eps(filepath):
        excel_timeids = re.findall(regex_en_timeid_q, str(df.loc[6]))
        excel_years = list(map(lambda x: int(x.split("/")[0]), excel_timeids))
        excel_months = list(map(lambda x: int(x.split("/")[1]), excel_timeids))
        excel_steps = [abs(a - b) for a, b in pairwise(excel_months)]
        excel_step = max(excel_steps, key=excel_steps.count)
        excel_last_quarter = ((excel_months[-1] - 1) // 3) + 1

        steptypes = {1: "monthly", 3: "quarterly", 0: "yearly"}
        if steptypes[excel_step] != time_type:
            print("unmatch time :", filepath)
            return True

        if time_type == "monthly" and last_month != False:
            if str(last_month) != str(excel_months[-1]):
                print("old data monthly :", filepath, excel_months[-1])
                return True

        if time_type == "quarterly" and last_quarter != False:
            if str(last_quarter) != str(excel_last_quarter):
                print("old data quarterly :", filepath, excel_last_quarter)
                return True

        if time_type == "yearly" and last_year != False:
            if str(last_year) != str(excel_years[-1]):
                print("old data yearly :", filepath, excel_years[-1])
                return True

        return False


def is_stock_file_not_match_author_type(df, filepath, per_article_type, stock_name):
    """Returns True if the file author or type does not match the expected values, False otherwise"""

    if not_pe_opt_eps(filepath):
        author = "Pouya Finance"
        excel_author = df["Unnamed: 1"][0]
        excel_type = df["Unnamed: 1"][4]
        excel_token = (df["Unnamed: 1"][3]).replace("\u200c", "").split("-")[0]

        if excel_author != author:
            print("not bourseview : ", filepath)
            return True

        if excel_type != per_article_type:
            print("unmatch type : ", filepath, per_article_type)
            return True

        if excel_token != wl_prod[stock_name]["token"]:
            print("unmatch name :", filepath)
            return True

        return False


def check_stock_files(
    stock_name, last_year=False, last_quarter=False, last_month=False, delete=False
):
    failed = []

    for filepath in essential_stock_files(stock_name):
        if Path(filepath).exists():
            try:
                df = pd.read_excel(filepath)
                article_type, per_article_type, time_type = filepath_info(filepath)
                if any(
                    [
                        is_stock_file_large_size(filepath),
                        is_stock_file_empty(df, filepath),
                        is_stock_file_not_match_time(
                            df, filepath, time_type, last_year, last_quarter, last_month
                        ),
                        is_stock_file_not_match_author_type(
                            df, filepath, per_article_type, stock_name
                        ),
                    ]
                ):
                    failed.append(filepath)

            except Exception as err:
                print(err, filepath)
                failed.append(filepath)

    for i in set(failed):
        print(i)
        if delete:
            os.remove(i)
