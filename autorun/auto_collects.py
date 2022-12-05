import os
import re
import sys
import shutil
import platform
import pandas as pd
import win32com.client as win32

from time import sleep
from pathlib import Path
from datetime import datetime
from persiantools.jdatetime import JalaliDate
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from autorun.basic_modules import to_digits, best_table_id
from statics.secrets import bourseview_user, bourseview_pass
from statics.driver_setup import (
    driver_options,
    driver_capabilities,
    break_time,
    wait_time,
)
from statics.setting import (
    DB,
    WINDOWS_FIREFOX_DRIVER_PATH,
    LINUX_FIREFOX_DRIVER_PATH,
    watchlist,
    first_day,
    last_day,
    regex_per_timeid_y,
)


# create driver for Linux and Windows
if platform.system() == "Linux":
    driver_options.set_preference("browser.download.dir", DB)
    driver = webdriver.Firefox(
        executable_path=LINUX_FIREFOX_DRIVER_PATH,
        options=driver_options,
        capabilities=driver_capabilities,
    )

if platform.system() == "Windows":
    driver_options.set_preference("browser.download.dir", DB.replace("/", "\\"))
    driver = webdriver.Firefox(
        executable_path=WINDOWS_FIREFOX_DRIVER_PATH,
        options=driver_options,
        capabilities=driver_capabilities,
    )

driver.implicitly_wait(wait_time)
driver.maximize_window()


def create_database_structure():
    """create database folders based on watchlist"""

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


def check_stock_files(stock_name):
    """check sanity,deficiency,unnecessary data"""

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

    dirs, files = list_stock_files(stock_name)

    def ignoreable(x):
        # not unnecessary and not deficiency
        if (
            "/detail_trade" in x
            or "/analyse" in x
            or "eps.xlsx" in x
            or "opt.xlsx" in x
            or "forward.xlsx" in x
        ):
            return True
        else:
            return False

    # all files
    for file in files:
        # get creation time of file
        t = JalaliDate(datetime.fromtimestamp(os.path.getctime(file)))
        if not ignoreable(file):
            if file not in base_files:
                # delete unnecessary files
                print("unnecessary file : ", file)
                os.remove(file)

            else:
                # check sanity of bourseview excels
                try:
                    df = pd.read_excel(file)
                    file_types = {
                        "balancesheet": "Balance Sheet",
                        "income": "Income Statements",
                        "cashflow": "Cash Flow",
                        "product": "تولید و فروش",
                        "cost": "بهای تمام شده",
                        "official": "هزینه های عمومی و اداری",
                        "pe": "تاریخچه قیمت",
                    }
                    file_type = file.split("/")[6]

                    if df["Unnamed: 1"][0] != "Pouya Finance":
                        print("not bourseview : ", file)

                    if df["Unnamed: 1"][4] != file_types[file_type]:
                        print("unmatch type : ", file)

                except:
                    print("old format : ", file)

    for dir in dirs:
        if not ignoreable(dir) and len(os.listdir(dir)) == 0:
            # delete empty folders
            os.rmdir(dir)

    # needed files
    for file in base_files:
        if not ignoreable(file) and not Path(file).exists():
            # show deficiency files
            print("deficiency : ", file)


def to_useful_excel(file_path):
    """convert bourseview excel to new excel that pandas use it"""
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
    sleep(2 * break_time)
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

    sleep(2 * break_time)
    if platform.system() == "Windows":
        to_useful_excel(new_path)


def codal_login():
    """login into codal"""

    # get page
    driver.get("https://www.codal.ir")
    sleep(break_time)

    # select 'jostojoye etelaye'
    driver.find_element(By.XPATH, '//*[@id="aSearch"]').click()
    sleep(break_time)


def codal_search(stock_name):
    """search stock in codal"""

    # click searck button
    driver.find_element(
        By.XPATH,
        "//*[@id='collapse-search-1']/div[2]/div[1]/div/div",
    ).click()
    sleep(break_time)

    # send stock name
    driver.find_element(By.XPATH, '//*[@id="txtSymbol"]').clear()
    sleep(break_time)

    driver.find_element(By.XPATH, '//*[@id="txtSymbol"]').send_keys(
        watchlist[stock_name]["name"]
    )
    sleep(break_time)

    # select first choice
    driver.find_element(
        By.XPATH,
        "//*[@id='ui-select-choices-row-0-0']/div",
    ).click()
    sleep(break_time)


def codal_eps(stock_name, n=5):
    """create eps.xlsx"""

    try:
        codal_search(stock_name)
        # click 'davat majamea'
        driver.find_element(
            By.XPATH,
            "//*[@id='reportType']/option[7]",
        ).click()
        sleep(break_time)

        # select 'nooe etelaye'
        driver.find_element(
            By.XPATH,
            "//*[@id='collapse-search-1']/div[2]/div[5]/div/div",
        ).click()
        sleep(break_time)

        # click 'tasmimat majmae omomi saliyane'
        driver.find_element(
            By.XPATH,
            "//*[@id='ui-select-choices-row-1-4']/div/div",
        ).click()
        sleep(break_time)

        # click 'jostojo'
        driver.find_element(
            By.XPATH,
            "//*[@id='aspnetForm']/div[3]/div[1]/div[1]/div[2]/div[1]/div/div[3]/div[1]/input",
        ).click()
        sleep(break_time)

        links = []
        years = []

        # links and years elements
        links_xpath = "//a[@class = 'letter-title ng-binding ng-scope']"

        for el in driver.find_elements(By.XPATH, links_xpath):

            link_timeids = re.findall(regex_per_timeid_y, el.text)

            if str(to_digits(link_timeids[0]))[:4] not in years:

                if "اعلام تنفس" not in el.text:
                    links.append(el.get_attribute("href"))
                    years.append(str(to_digits(link_timeids[0]))[:4])

        # set user n for lables
        links = links[:n]
        years = years[:n]

        # find eps,dps,capital
        dates = []
        eps = []
        dps = []
        capital = []

        driver.implicitly_wait(10)

        for link in links:
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(link)

            date_xpath = "//*[@id='tblAssembly']/tbody/tr[2]/td/bdo[2]"
            eps_xpath = '//*[@id="ucAssemblyPRetainedEarning_grdAssemblyProportionedRetainedEarning_ctl17_Span1"]'
            dps_xpath = '//*[@id="ucAssemblyPRetainedEarning_grdAssemblyProportionedRetainedEarning_ctl18_Span1"]'
            capital_xpath = '//*[@id="ucAssemblyPRetainedEarning_grdAssemblyProportionedRetainedEarning_ctl19_Span1"]'

            try:
                dates.append(driver.find_element(By.XPATH, date_xpath).text)
                eps.append(to_digits(driver.find_element(By.XPATH, eps_xpath).text))
                dps.append(to_digits(driver.find_element(By.XPATH, dps_xpath).text))
                capital.append(
                    to_digits(driver.find_element(By.XPATH, capital_xpath).text)
                )

            except:
                dates.append("-")
                eps.append("-")
                dps.append("-")
                capital.append("-")

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        sleep(2 * break_time)
        driver.implicitly_wait(wait_time)

        # set user n for data
        dates = dates[:n]
        eps = eps[:n]
        dps = dps[:n]
        capital = capital[:n]

        # create df of (years,dates,eps,dps,capital,capital_now)
        df = pd.DataFrame()
        df["year"] = years
        df["date"] = dates
        df["EPS"] = eps
        df["DPS"] = dps
        df["capital"] = capital
        df["capital_now"] = [capital[0] for i in range((len(capital)))]

        # export eps.xlsx
        df.to_excel(
            f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/eps.xlsx",
            index=False,
        )

    except Exception as err:
        print(f"cant download eps {stock_name} : {err}")


def codal_statement(stock_name):
    """create 3 files : income,balancesheet,cashflow"""

    codal_search(stock_name)

    # click 'sorathaye mali'
    driver.find_element(
        By.XPATH,
        "//*[@id='reportType']/option[2]",
    ).click()
    sleep(break_time)

    # select 'nooe etelaye'
    driver.find_element(
        By.XPATH,
        "//*[@id='collapse-search-1']/div[2]/div[5]/div/div",
    ).click()
    sleep(break_time)

    # click 'mian dorea'
    driver.find_element(
        By.XPATH,
        "//*[@id='ui-select-choices-row-1-1']/div/div",
    ).click()
    sleep(break_time)

    # disable 'zirmajmoeha'
    driver.find_element(
        By.XPATH,
        "//*[@id='collapse-search-other']/div/div[6]/div[1]/div",
    ).click()
    sleep(break_time)

    # click 'jostojo'
    driver.find_element(
        By.XPATH,
        "//*[@id='aspnetForm']/div[3]/div[1]/div[1]/div[2]/div[1]/div/div[3]/div[1]/input",
    ).click()
    sleep(break_time)

    # create list of stocks_links and stocks_timeid
    stocks_links = []
    stocks_timeid = []
    for element in driver.find_elements(
        By.XPATH, "//a[@class = 'letter-title ng-binding ng-scope']"
    ):
        stocks_links.append(element.get_attribute("href"))

        base_string = element.text
        all_timeids = re.findall(
            "[۰۱۲۳۴۵۶۷۸۹]{4}/[۰۱۲۳۴۵۶۷۸۹]{2}/[۰۱۲۳۴۵۶۷۸۹]{2}", base_string
        )

        if "نشده" in base_string:
            if "اصلاحیه" in base_string:
                stocks_timeid.append(str(to_digits(all_timeids[0])) + "EHN")

            if "تلفیقی" in base_string:
                stocks_timeid.append(str(to_digits(all_timeids[0])) + "ETHN")

            else:
                stocks_timeid.append(str(to_digits(all_timeids[0])) + "HN")

        else:
            if "اصلاحیه" in base_string:
                stocks_timeid.append(str(to_digits(all_timeids[0])) + "EH")

            if "تلفیقی" in base_string:
                stocks_timeid.append(str(to_digits(all_timeids[0])) + "ETH")

            else:
                stocks_timeid.append(str(to_digits(all_timeids[0])) + "H")

    # create codal excel
    page_options = {1: "income", 0: "balancesheet", 9: "cashflow"}

    for page_option in list(page_options.keys()):
        df_list = []

        for stock_link in stocks_links:
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(stock_link + f"&sheetId={page_option}")

            page = driver.page_source

            data_tables = pd.read_html(page)
            t_id = best_table_id(data_tables)
            df = data_tables[t_id]

            df_list.append(df)

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        Excelwriter = pd.ExcelWriter(
            f"{DB}/{stock_name}_{page_options[page_option]}.xlsx",
            engine="xlsxwriter",
        )

        for i, df in enumerate(df_list):
            df.to_excel(Excelwriter, sheet_name=stocks_timeid[i])

        Excelwriter.save()


def tse_buy_sell_volume(stock_name):
    """return volume of buyer and seller"""
    driver.get("http://www.tsetmc.com/")
    sleep(break_time)

    # select search
    driver.find_element(By.XPATH, '//*[@id="search"]').click()
    sleep(break_time)

    # send stock name
    driver.find_element(By.XPATH, '//*[@id="SearchKey"]').send_keys(
        watchlist[stock_name]["token"]
    )
    sleep(break_time)

    # click blanck page
    driver.find_element(By.XPATH, '//*[@id="ModalWindowInner1"]').click()
    sleep(break_time)

    # select first choice
    driver.find_element(
        By.XPATH,
        "/html/body/div[5]/section/div/div/div/div[2]/table/tbody/tr[1]/td[1]/a",
    ).click()
    sleep(10 * break_time)

    # find 'kharid va forosh'
    buy_sell = (
        driver.find_element(By.XPATH, '//*[@id="Section_bestlimit"]').text
    ).split()

    # export excel
    df = pd.DataFrame([buy_sell[a : a + 6] for a in range(0, len(buy_sell), 6)])
    df.to_excel(
        f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/buy_sell_volume.xlsx",
        index=False,
    )


def bourseview_login():
    """login into bourseview panel"""

    # get page
    driver.get("https://www.bourseview.com/home/#/account/login")
    sleep(break_time)

    # go to login page
    driver.find_element(
        By.XPATH, "//a[@class='web-app-log-in-mobile web-app-log-in-btn']"
    ).click()
    sleep(break_time)

    # send username
    driver.find_element(By.XPATH, "//input[@id='Username']").send_keys(bourseview_user)
    sleep(break_time)

    # send password
    driver.find_element(By.XPATH, "//input[@id='Password']").send_keys(bourseview_pass)
    sleep(break_time)

    # login
    driver.find_element(By.XPATH, "//*[@id='submit_btn']").click()
    sleep(6 * break_time)

    try:
        # block pop-ups
        driver.find_element(By.XPATH, "//*[@id='dialog_1']/div[1]/div[1]/span").click()
        sleep(break_time)

    except:
        pass


def bourseview_search(stock_name):
    """search stock in bourseview"""

    # search stock name
    driver.find_element(By.XPATH, "//*[@id='input-0']").clear()
    sleep(break_time)
    driver.find_element(By.XPATH, "//*[@id='input-0']").send_keys(
        watchlist[stock_name]["token"]
    )
    sleep(break_time)

    # select first choice
    driver.find_element(
        By.XPATH,
        "/html/body/md-virtual-repeat-container/div/div[2]/ul/li[1]/md-autocomplete-parent-scope/div/div/div[1]",
    ).click()
    sleep(2 * break_time)


def bourseview_balancesheet(stock_name, y=10, q=10):
    """download 2 files : yearly,quarterly"""
    try:
        bourseview_search(stock_name)

        # select 'tarazname'
        driver.find_element(By.XPATH, "//*[@id='stocks-sub-menu']/ul/li[2]").click()
        sleep(break_time)

        # click blanck page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # download 2 excels
        for time_type in ("yearly", "quarterly"):

            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            driver.find_element(By.XPATH, f"//option[@value='{time_type}']").click()
            sleep(2 * break_time)

            # select 'dore'
            driver.find_element(By.XPATH, f"//option[@value='{n}']").click()
            sleep(4 * break_time)

            # click download excel
            driver.find_element(
                By.XPATH, "//*[@id='new-balance-sheet-grid']/div/div[1]/span[2]/span"
            ).click()
            sleep(2 * break_time)

            # replace last file
            move_last_file(
                f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/balancesheet/{time_type}.xlsx"
            )
    except Exception as err:
        print(f"cant download balancesheet {stock_name} : {err}")


def bourseview_income_statement(stock_name, y=10, q=10):
    """download 4 files : yearly,quarterly,rial,dollar"""
    try:
        bourseview_search(stock_name)

        # select 'sood va zian'
        driver.find_element(By.XPATH, "//*[@id='stocks-sub-menu']/ul/li[3]").click()
        sleep(break_time)

        # click blank page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # download 4 excels
        for time_type in ("yearly", "quarterly"):

            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            driver.find_element(By.XPATH, f"//option[@value='{time_type}']").click()
            sleep(2 * break_time)

            # select 'dore'
            driver.find_element(By.XPATH, f"//option[@value='{n}']").click()
            sleep(2 * break_time)

            for money_type in ("IRR", "USDf"):

                # click 'rial','dollar azad'
                driver.find_element(
                    By.XPATH, f"//option[@value='{money_type}']"
                ).click()
                sleep(6 * break_time)

                # click download excel
                driver.find_element(
                    By.XPATH,
                    "//*[@id='new-income-statement-grid']/div/div[1]/span[2]/span",
                ).click()
                sleep(2 * break_time)

                # replace last file
                money_options = {"IRR": "rial", "USDf": "dollar"}
                move_last_file(
                    f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/income/{time_type}/{money_options[money_type]}.xlsx"
                )
    except Exception as err:
        print(f"cant download incomestatement {stock_name} : {err}")


def bourseview_cashflow(stock_name, y=10, q=10):
    """download 2 files : yearly,quarterly"""

    try:
        bourseview_search(stock_name)

        # select 'jaryan vojoh naghd'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-sub-menu']/ul/li[4]",
        ).click()
        sleep(break_time)

        # click blanck page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # download 2 file
        for time_type in ("yearly", "quarterly"):

            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            driver.find_element(By.XPATH, f"//option[@value='{time_type}']").click()
            sleep(2 * break_time)

            # select 'dore'
            driver.find_element(By.XPATH, f"//option[@value='{n}']").click()
            sleep(4 * break_time)

            # click download excel
            driver.find_element(
                By.XPATH, "//*[@id='new-cash-flow-grid']/div/div[1]/span[2]/span"
            ).click()
            sleep(2 * break_time)

            # replace last file
            move_last_file(
                f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/cashflow/{time_type}.xlsx"
            )
    except Exception as err:
        print(f"cant download cashflow {stock_name} : {err}")


def bourseview_product_revenue(stock_name, y=10, q=10, m=50):
    """create 6 files : (yearly,quarterly,monthly) (seprated)
    <y = year : 5,10,20,50>
    <q = quarterly : 5,10,20,50>
    <m = month : 5,10,20,50>"""

    try:

        def dl_excels(new_name=""):

            # download seprated excels
            for time_type in ("yearly", "quarterly", "monthly"):

                if time_type == "yearly":
                    n = y

                if time_type == "quarterly":
                    n = q

                if time_type == "monthly":
                    n = m

                # select 'nooe'
                driver.find_element(By.XPATH, f"//option[@value='{time_type}']").click()
                sleep(2 * break_time)

                # select 'dore'
                driver.find_element(By.XPATH, f"//option[@value='{n}']").click()
                sleep(4 * break_time)

                # click download excel
                driver.find_element(
                    By.XPATH,
                    "//*[@id='grid']/div/div[1]/span[2]/span",
                ).click()
                sleep(2 * break_time)

                # replace last file
                move_last_file(
                    f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/product/{time_type}{new_name}.xlsx"
                )

        bourseview_search(stock_name)

        # select 'tolid va frosh'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-sub-menu']/ul/li[8]",
        ).click()
        sleep(break_time)

        # click blanck page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # download base excels
        dl_excels()

        # click 'tafkik dakheli,khareji'
        driver.find_element(
            By.XPATH,
            "//*[@id='grid']/div/div[1]/span[1]/div[3]",
        ).click()
        sleep(4 * break_time)

        # download seprated excels
        dl_excels("_seprated")

        # disable 'tafkik dakheli,khareji'
        driver.find_element(
            By.XPATH,
            "//*[@id='grid']/div/div[1]/span[1]/div[3]",
        ).click()
        sleep(2 * break_time)
    except Exception as err:
        print(f"cant download product {stock_name} : {err}")


def bourseview_cost(stock_name, y=10, q=10):
    """create 2 excel : yearly,quarterly
    <y = year : 5,10,20,50>
    <q = quarterly : 5,10,20,50>"""

    try:
        bourseview_search(stock_name)

        # select 'bahaye tamam shode'
        driver.find_element(By.XPATH, "//*[@id='stocks-sub-menu']/ul/li[13]").click()
        sleep(break_time)

        # click blanck page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # download excel
        for time_type in ("yearly", "quarterly"):

            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            driver.find_element(By.XPATH, f"//option[@value='{time_type}']").click()
            sleep(2 * break_time)

            # select 'dore'
            driver.find_element(By.XPATH, f"//option[@value='{n}']").click()
            sleep(4 * break_time)

            # click 'hameye' data
            driver.find_element(
                By.XPATH, "//*[@id='grid-cogs']/div/div[4]/div/div[1]"
            ).click()
            sleep(2 * break_time)

            # go top of page
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.CONTROL + Keys.HOME)
            sleep(4 * break_time)

            # click download excel
            driver.find_element(
                By.XPATH, "//*[@id='grid-cogs']/div/div[2]/span[2]/span"
            ).click()
            sleep(2 * break_time)

            # replace last file
            move_last_file(
                f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/cost/{time_type}.xlsx"
            )
    except Exception as err:
        print(f"cant download cost {stock_name} : {err}")


def bourseview_official(stock_name, y=10, q=10):
    """create 2 excel : yearly,quarterly
    <y = year : 5,10,20,50>
    <q = quarterly : 5,10,20,50>"""

    try:
        bourseview_search(stock_name)

        # select 'hazine haye omomi edari'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-sub-menu']/ul/li[16]",
        ).click()
        sleep(break_time)

        # click blanck page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # 'salane','fasli'
        for time_type in ("yearly", "quarterly"):

            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            driver.find_element(By.XPATH, f"//option[@value='{time_type}']").click()
            sleep(2 * break_time)

            # select 'dore'
            driver.find_element(By.XPATH, f"//option[@value='{n}']").click()
            sleep(6 * break_time)

            # click download excel
            driver.find_element(
                By.XPATH, "//*[@id='grid']/div/div[2]/span[2]/span"
            ).click()
            sleep(2 * break_time)

            # replace last file
            move_last_file(
                f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/official/{time_type}.xlsx"
            )
    except Exception as err:
        print(f"cant download official {stock_name} : {err}")


def bourseview_price_history(stock_name, start=first_day, end=last_day):
    """download pe.xlsx
    <start : 1390/01/01>
    <end : 1400/01/01>"""
    try:
        bourseview_search(stock_name)
        # select 'tarikhche gheimat'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-sub-menu']/ul/li[21]",
        ).click()

        sleep(4 * break_time)

        # click blank page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # send 'tarikh shoroea'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/input",
        ).clear()
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/input",
        ).send_keys(start)
        sleep(break_time)

        # send 'tarikh payan'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/div[3]/div[1]/input",
        ).clear()
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/div[3]/div[1]/input",
        ).send_keys(end)
        sleep(break_time)

        # click 'namayesh gheimat'
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/button",
        ).click()
        sleep(2 * break_time)

        # click download excel
        driver.find_element(
            By.XPATH,
            "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[2]/div/div/div[2]/div",
        ).click()
        sleep(2 * break_time)

        # replace last file
        move_last_file(
            f"{DB}/industries/{watchlist[stock_name]['indus']}/{stock_name}/pe/pe.xlsx"
        )
    except:
        print(f"cant download price history of {stock_name}")


def bourseview_macro(start=first_day, end=last_day):
    """download macro.xlsx
    <start : 1390/01/01>
    <end : 1400/01/01>"""
    try:
        # select 'dadehaye kalan'
        driver.find_element(
            By.XPATH,
            "//*[@id='step3']/li[9]",
        ).click()
        sleep(break_time)

        # select 'shakhes boors iran'
        driver.find_element(
            By.XPATH,
            "//*[@id='macro-history-select-irex-wrapper']/span",
        ).click()
        sleep(break_time)

        # click 'arzesh moamelat'
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[3]"
        ).click()
        sleep(break_time)

        # click p/e ttm
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[6]"
        ).click()
        sleep(break_time)

        # click p/d
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[8]"
        ).click()
        sleep(break_time)

        # select 'saham shakhes boors'
        driver.find_element(
            By.XPATH,
            "//*[@id='macro-history-select-tedpix-wrapper']/span",
        ).click()
        sleep(break_time)

        # click 'arzesh khales vorod haghighi'
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[20]"
        ).click()
        sleep(break_time)

        # select 'nerkh arz'
        driver.find_element(
            By.XPATH,
            "//*[@id='macro-history-select-currency-wrapper']/span",
        ).click()
        sleep(break_time)

        # click dollar 'nima'
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[1]"
        ).click()
        sleep(break_time)

        # click dollar 'azad'
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[2]"
        ).click()
        sleep(break_time)

        # select 'motoghayerhaye poli'
        driver.find_element(
            By.XPATH,
            "//*[@id='macro-history-select-economics-wrapper']/span",
        ).click()
        sleep(break_time)

        # click 'miangin nerkh bedon risk'
        driver.find_element(
            By.XPATH, "/html/body/span/span/span/ul/li/ul/li[2]"
        ).click()
        sleep(break_time)

        # click download excel
        driver.find_element(
            By.XPATH,
            "/html/body/div[2]/div/div/div/div/div[3]/div[3]/div/div[1]/div/div/div[3]/div/span",
        ).click()
        sleep(break_time)

        # click 'entekhab baze delkhah'
        driver.find_element(
            By.XPATH,
            "//*[@id='myModal']/div/div/div[2]/div/button",
        ).click()
        sleep(break_time)

        # send 'tarikh shoroe'
        driver.find_element(
            By.XPATH,
            "//*[@id='collapseExample']/div/div/div[1]/div[1]/input",
        ).send_keys(start)
        sleep(break_time)

        # send 'tarikh payan'
        driver.find_element(
            By.XPATH,
            "//*[@id='collapseExample']/div/div/div[3]/div[1]/input",
        ).send_keys(end)
        sleep(break_time)

        # click download button
        driver.find_element(
            By.XPATH,
            "//*[@id='myModal']/div/div/div[3]/button",
        ).click()
        sleep(2 * break_time)

        # click blank page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # replace last file
        move_last_file(f"{DB}/macro/macro.xlsx")
    except Exception as err:
        print(f"cant download macro data : {err}")


def download_database_files(lst=list(watchlist.keys()), y=10, q=10, m=50):
    '''download all of stock files'''
    create_database_structure()
    bourseview_login()
    for i in lst:
        bourseview_balancesheet(i, y, q)
        bourseview_income_statement(i, y, q)
        bourseview_cashflow(i, y, q)
        bourseview_product_revenue(i, y, q, m)
        bourseview_cost(i, y, q)
        bourseview_official(i, y, q)
        bourseview_price_history(i)

    codal_login()
    for i in lst:
        codal_eps(i)
