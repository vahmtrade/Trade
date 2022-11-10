import os
from pathlib import Path
from statics.setting import DB, watchlist


def create_database_structure():
    """create database folders based on watchlist"""

    for stock, info in watchlist.items():

        for i in (
            "balancesheet",
            "income",
            "cashflow",
            "product",
            "cost",
            "official",
            "pe",
            "analyse",
            'detail_trade',
        ):

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
    stock_files = []
    for path, subdirs, files in os.walk(
        f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}"
    ):
        for name in files:
            stock_files.append(os.path.join(path, name).replace("\\", "/"))

    return stock_files


def update_stock_files(stock):
    stock_folder = f"{DB}/industries/{watchlist[stock]['indus']}/{stock}"

    # files that must every stock have it
    base_files = [
        f"{stock_folder}/balancesheet/quarterly.xlsx",
        f"{stock_folder}/balancesheet/yearly.xlsx",
        f"{stock_folder}/cashflow/quarterly.xlsx",
        f"{stock_folder}/cashflow/yearly.xlsx",
        f"{stock_folder}/cost/quarterly.xlsx",
        f"{stock_folder}/cost/yearly.xlsx",
        f"{stock_folder}/income/quarterly/dollar.xlsx",
        f"{stock_folder}/income/quarterly/rial.xlsx",
        f"{stock_folder}/income/yearly/dollar.xlsx",
        f"{stock_folder}/income/yearly/rial.xlsx",
        f"{stock_folder}/official/quarterly.xlsx",
        f"{stock_folder}/official/yearly.xlsx",
        f"{stock_folder}/pe/pe.xlsx",
        f"{stock_folder}/product/monthly.xlsx",
        f"{stock_folder}/product/monthly_seprated.xlsx",
        f"{stock_folder}/product/quarterly.xlsx",
        f"{stock_folder}/product/quarterly_seprated.xlsx",
        f"{stock_folder}/product/yearly.xlsx",
        f"{stock_folder}/product/yearly_seprated.xlsx",
        f"{stock_folder}/eps.xlsx",
        f"{stock_folder}/opt.xlsx",
    ]

    # show deficiency files
    print(100 * "#")
    print(f"<<< DEFICIENCY {stock.upper()} >>>")
    for b in base_files:
        if Path(b).exists() == False:
            print(b.replace("\\", "/"))

    # remove unnecessary files
    print(100 * "-")
    print(f"<<< UNNECESSARY {stock.upper()} >>>")
    all_files = list_stock_files(stock)
    for a in all_files:
        if a not in base_files:
            print(a)
            os.remove(a)


def update_database():
    for stock in list(watchlist.keys()):
        update_stock_files(stock)


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
