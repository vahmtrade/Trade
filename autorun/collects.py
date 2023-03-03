import re
import platform
import pandas as pd

from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    presence_of_element_located as presence,
    element_to_be_clickable as clickable,
    staleness_of as staleness,
    invisibility_of_element as invisibility,
)

from statics.setting import *
from statics.secrets import *
from statics.dvset import *
from preprocess.basic import *

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


webwait = WebDriverWait(driver, wait_time)
driver.maximize_window()


def ime_physical(start=month_ago, end=today_10char):
    webwait = WebDriverWait(driver, 4 * wait_time)
    # get page
    driver.get("https://ime.co.ir/offer-stat.html")
    sleep(break_time)

    # send start day
    first = "//*[@id='ctl05_ReportsHeaderControl_FromDate']"
    webwait.until(presence((By.XPATH, first)))
    driver.find_element(By.XPATH, first).clear()
    driver.find_element(By.XPATH, first).send_keys(start)
    sleep(break_time)

    # send end day
    last = "//*[@id='ctl05_ReportsHeaderControl_ToDate']"
    webwait.until(presence((By.XPATH, last)))
    driver.find_element(By.XPATH, last).clear()
    driver.find_element(By.XPATH, last).send_keys(end)
    sleep(break_time)

    # click 'satrha'
    rows = "//*[@id='AmareMoamelatGrid']/div[1]/div[1]/div[1]/div[1]/button"
    webwait.until(clickable((By.XPATH, rows)))
    driver.find_element(By.XPATH, rows).click()
    sleep(break_time)

    # click 'tarikh moamele'
    t = "//*[@id='AmareMoamelatGrid']/div[1]/div[1]/div[1]/div[1]/ul/li[16]/label/input"
    webwait.until(clickable((By.XPATH, t)))
    driver.find_element(By.XPATH, t).click()
    sleep(break_time)

    # click 'namayesh'
    show = "//*[@id='FillGrid']"
    webwait.until(clickable((By.XPATH, show)))
    driver.find_element(By.XPATH, show).click()
    loading = "/html/body/div[2]/div/div/div[1]/div/div[2]"
    webwait.until(staleness(driver.find_element(By.XPATH, loading)))
    sleep(4 * break_time)

    # click 'Export data'
    export = "//*[@id='AmareMoamelatGrid']/div[1]/div[1]/div[1]/div[2]/button"
    webwait.until(clickable(driver.find_element(By.XPATH, export)))
    driver.find_element(By.XPATH, export).click()
    sleep(break_time)

    # click CSV
    csv = "//*[@id='AmareMoamelatGrid']/div[1]/div[1]/div[1]/div[2]/ul/li[3]"
    webwait.until(clickable((By.XPATH, csv)))
    driver.find_element(By.XPATH, csv).click()
    sleep(2 * break_time)

    # replace last file
    new_path = f"{MACROPATH}/physical.csv"
    move_last_file(new_path)
    sleep(2 * break_time)

    # save as excel file
    save_as_file(new_path, "xls")
    sleep(2 * break_time)


def codal_login():
    """login into codal"""
    try:
        # get page
        driver.get("https://www.codal.ir")
        sleep(break_time)

        # select 'jostojoye etelaye'
        search = '//*[@id="aSearch"]'
        webwait.until(clickable((By.XPATH, search)))
        driver.find_element(By.XPATH, search).click()
        sleep(break_time)

    except Exception as err:
        print(f"cant login into codal : {err}")


def codal_search(name):
    """search stock in codal"""
    try:
        # click searck button
        search = "//*[@id='collapse-search-1']/div[2]/div[1]/div/div"
        webwait.until(clickable((By.XPATH, search)))
        driver.find_element(By.XPATH, search).click()
        sleep(break_time)

        # send stock name
        send = '//*[@id="txtSymbol"]'
        webwait.until(presence((By.XPATH, send)))
        driver.find_element(By.XPATH, send).clear()
        driver.find_element(By.XPATH, send).send_keys(wl_prod[name]["name"])
        sleep(break_time)

        exceptions = {"simorgh": 2}
        if name in exceptions:
            # select exceptions choice
            exception = f"//*[@id='ui-select-choices-row-0-{exceptions[name]}']"
            webwait.until(clickable((By.XPATH, exception)))
            driver.find_element(By.XPATH, exception).click()
            sleep(break_time)

        else:
            # select first choice
            first = "//*[@id='ui-select-choices-row-0-0']"
            webwait.until(clickable((By.XPATH, first)))
            driver.find_element(By.XPATH, first).click()
            sleep(break_time)

        sleep(break_time)

    except Exception as err:
        print(f"cant codal search {name} : {err}")


def codal_eps(name, n=5):
    """create eps"""
    try:
        # click 'davat majamea'
        call = "//*[@id='reportType']/option[7]"
        webwait.until(clickable((By.XPATH, call)))
        driver.find_element(By.XPATH, call).click()
        sleep(break_time)

        # select 'nooe etelaye'
        notice = "//*[@id='collapse-search-1']/div[2]/div[5]/div/div"
        webwait.until(clickable((By.XPATH, notice)))
        driver.find_element(By.XPATH, notice).click()
        sleep(break_time)

        # click 'tasmimat majmae omomi saliyane'
        choice = "//*[@id='ui-select-choices-row-1-4']/div/div"
        webwait.until(clickable((By.XPATH, choice)))
        driver.find_element(By.XPATH, choice).click()
        sleep(break_time)

        # click 'jostojo'
        search = "//*[@id='aspnetForm']/div[3]/div[1]/div[1]/div[2]/div[1]/div/div[3]/div[1]/input"
        webwait.until(clickable((By.XPATH, search)))
        driver.find_element(By.XPATH, search).click()
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

        for link in links:
            # open link in new page
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

        # select user n of data
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

        # export eps
        path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['eps']}"
        df.to_excel(path, index=False)

    except Exception as err:
        print(f"cant download eps {name} : {err}")


def codal_statement(name):
    """create 3 files : income,balancesheet,cashflow"""

    # click 'sorathaye mali'
    state = "//*[@id='reportType']/option[2]"
    webwait.until(clickable((By.XPATH, state)))
    driver.find_element(By.XPATH, state).click()
    sleep(break_time)

    # select 'nooe etelaye'
    notice = "//*[@id='collapse-search-1']/div[2]/div[5]/div/div"
    webwait.until(clickable((By.XPATH, notice)))
    driver.find_element(By.XPATH, notice).click()
    sleep(break_time)

    # click 'mian dorea'
    mid = "//*[@id='ui-select-choices-row-1-1']/div/div"
    webwait.until(clickable((By.XPATH, mid)))
    driver.find_element(By.XPATH, mid).click()
    sleep(break_time)

    # disable 'zirmajmoeha'
    subset = "//*[@id='collapse-search-other']/div/div[6]/div[1]/div"
    webwait.until(clickable((By.XPATH, subset)))
    driver.find_element(By.XPATH, subset).click()
    sleep(break_time)

    # click 'jostojo'
    search = "//*[@id='aspnetForm']/div[3]/div[1]/div[1]/div[2]/div[1]/div/div[3]/div[1]/input"
    webwait.until(clickable((By.XPATH, search)))
    driver.find_element(By.XPATH, search).click()
    sleep(break_time)

    # create list of stocks_links and stocks_timeid
    links_xpath = "//a[@class = 'letter-title ng-binding ng-scope']"
    stocks_links = []
    stocks_timeid = []

    for element in driver.find_elements(By.XPATH, links_xpath):
        stocks_links.append(element.get_attribute("href"))

        base_string = element.text
        all_timeids = re.findall(regex_per_timeid_y, base_string)

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
            # open link in new page
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(stock_link + f"&sheetId={page_option}")

            page = driver.page_source
            data_tables = pd.read_html(page)
            df = data_tables[best_table_id(data_tables)]
            df_list.append(df)

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        new_path = f"{DB}/{name}_{page_options[page_option]}.xlsx"
        Excelwriter = pd.ExcelWriter(new_path, engine="xlsxwriter")

        for i, df in enumerate(df_list):
            df.to_excel(Excelwriter, sheet_name=stocks_timeid[i])

        Excelwriter.save()


def bourseview_login():
    """login into bourseview panel"""

    try:
        # get page
        driver.get("https://www.bourseview.com/home/#/account/login")
        sleep(break_time)

        # go to login page
        page = "//a[@class='web-app-log-in-mobile web-app-log-in-btn']"
        webwait.until(clickable((By.XPATH, page)))
        driver.find_element(By.XPATH, page).click()
        sleep(break_time)

        # send username
        user = "//input[@id='Username']"
        webwait.until(presence((By.XPATH, user)))
        driver.find_element(By.XPATH, user).send_keys(bourseview_user)
        sleep(break_time)

        # send password
        passw = "//input[@id='Password']"
        webwait.until(presence((By.XPATH, passw)))
        driver.find_element(By.XPATH, passw).send_keys(bourseview_pass)
        sleep(break_time)

        # login
        login = "//*[@id='submit_btn']"
        webwait.until(clickable((By.XPATH, login)))
        driver.find_element(By.XPATH, login).click()
        sleep(12 * break_time)

        try:
            # block pop-ups
            block = "//*[@id='dialog_1']/div[1]/div[1]/span"
            driver.find_element(By.XPATH, block).click()
            sleep(break_time)

        except:
            pass

    except Exception as err:
        print(f"cant login into bourseview : {err}")


def bourseview_search(name):
    """search stock in bourseview"""

    try:
        # search stock name
        search = "//*[@id='input-0']"
        webwait.until(presence((By.XPATH, search)))
        driver.find_element(By.XPATH, search).clear()
        driver.find_element(By.XPATH, search).send_keys(wl_prod[name]["token"])
        sleep(break_time)

        # select first choice
        first = "/html/body/md-virtual-repeat-container/div/div[2]/ul/li[1]/md-autocomplete-parent-scope/div/div/div[1]"
        webwait.until(clickable((By.XPATH, first)))
        driver.find_element(By.XPATH, first).click()
        sleep(2 * break_time)

    except Exception as err:
        print(f"cant bourseview search {name} : {err}")


def bourseview_balancesheet(name, y=5, q=5, time_types=["yearly", "quarterly"]):
    """download 2 files : yearly,quarterly"""
    try:
        # select 'tarazname'
        balancesheet = "//*[@id='stocks-sub-menu']/ul/li[2]/a[2]"
        webwait.until(clickable((By.XPATH, balancesheet)))
        driver.find_element(By.XPATH, balancesheet).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "/html").click()
        loading_xpath = "//*[@id='new-balance-sheet-grid']/div/div[4]"
        loading = driver.find_element(By.XPATH, loading_xpath)
        webwait.until(invisibility(loading))
        sleep(4 * break_time)

        for time_type in time_types:
            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            balancesheet_type = f"//option[@value='{time_type}']"
            webwait.until(clickable((By.XPATH, balancesheet_type)))
            driver.find_element(By.XPATH, balancesheet_type).click()
            sleep(break_time)

            # select 'dore'
            balancesheet_count = f"//option[@value='{n}']"
            webwait.until(clickable((By.XPATH, balancesheet_count)))
            driver.find_element(By.XPATH, balancesheet_count).click()
            webwait.until(invisibility(loading))
            sleep(4 * break_time)

            # click download excel
            dl_btn = "//*[@id='new-balance-sheet-grid']/div/div[1]/span[2]/span"
            webwait.until(clickable((By.XPATH, dl_btn)))
            driver.find_element(By.XPATH, dl_btn).click()
            sleep(2 * break_time)

            # replace last file
            new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['balance'][time_type]}"
            move_last_file(new_path)
            sleep(2 * break_time)

            # open and ctrl + s excel file
            resave_excel(new_path)
            sleep(2 * break_time)

    except Exception as err:
        print(f"cant download balancesheet {name} : {err}")


def bourseview_income_statement(
    name,
    y=5,
    q=5,
    time_types=["yearly", "quarterly"],
    currency_types=["rial", "dollar"],
    report_types=["_cumulative", ""],
):
    """download 4 files : yearly,quarterly,rial,dollar"""

    def dl_excels(report_type=""):
        # select 'nooe'
        income_type = f"//option[@value='{time_type}']"
        webwait.until(clickable((By.XPATH, income_type)))
        driver.find_element(By.XPATH, income_type).click()
        sleep(break_time)

        # select 'dore'
        income_count = f"//option[@value='{n}']"
        webwait.until(clickable((By.XPATH, income_count)))
        driver.find_element(By.XPATH, income_count).click()
        webwait.until(invisibility(loading))
        sleep(break_time)

        if report_type == "_cumulative":
            # click 'gozaresh tajmiee'
            cumulative = "//*[@id='new-income-statement-grid']/div/div[1]/span[1]/div/span[1]/span[2]"
            webwait.until(clickable((By.XPATH, cumulative)))
            driver.find_element(By.XPATH, cumulative).click()
            webwait.until(invisibility(loading))
            sleep(break_time)

        for currency_type in currency_types:
            # click 'rial','dollar azad'
            currency_options = {"rial": "IRR", "dollar": "USDf"}
            currency = f"//option[@value='{currency_options[currency_type]}']"
            webwait.until(clickable((By.XPATH, currency)))
            driver.find_element(By.XPATH, currency).click()
            webwait.until(invisibility(loading))
            sleep(4 * break_time)

            # click download excel
            dl_btn = "//*[@id='new-income-statement-grid']/div/div[1]/span[2]/span"
            webwait.until(clickable((By.XPATH, dl_btn)))
            driver.find_element(By.XPATH, dl_btn).click()
            sleep(2 * break_time)

            # replace last file
            new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['income'][time_type][report_type][currency_type]}"
            move_last_file(new_path)
            sleep(2 * break_time)

            # open and ctrl + s excel file
            resave_excel(new_path)
            sleep(2 * break_time)

    try:
        # select 'sood va zian'
        income = "//*[@id='stocks-sub-menu']/ul/li[3]"
        webwait.until(clickable((By.XPATH, income)))
        driver.find_element(By.XPATH, income).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "//*[@id='overal_step2']/div[1]/div/div").click()
        loading_xpath = "//*[@id='new-income-statement-grid']/div/div[4]"
        loading = driver.find_element(By.XPATH, loading_xpath)
        sleep(4 * break_time)

        for time_type in time_types:
            if time_type == "yearly":
                n = y
                # download 2 excels
                dl_excels()

            if time_type == "quarterly":
                n = q
                for report_type in report_types:
                    # download 4 excels
                    dl_excels(report_type)

    except Exception as err:
        print(f"cant download incomestatement {name} : {err}")


def bourseview_cashflow(name, y=5, q=5, time_types=["yearly", "quarterly"]):
    """download 2 files : yearly,quarterly"""

    try:
        # select 'jaryan vojoh naghd'
        cashflow = "//*[@id='stocks-sub-menu']/ul/li[4]/a[2]"
        webwait.until(clickable((By.XPATH, cashflow)))
        driver.find_element(By.XPATH, cashflow).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "/html").click()
        loading_xpath = "//*[@id='new-cash-flow-grid']/div/div[4]"
        loading = driver.find_element(By.XPATH, loading_xpath)
        webwait.until(invisibility(loading))
        sleep(4 * break_time)

        # download 2 file
        for time_type in time_types:
            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            cashflow_type = f"//option[@value='{time_type}']"
            webwait.until(clickable((By.XPATH, cashflow_type)))
            driver.find_element(By.XPATH, cashflow_type).click()
            sleep(break_time)

            # select 'dore'
            cashflow_count = f"//option[@value='{n}']"
            webwait.until(clickable((By.XPATH, cashflow_count)))
            driver.find_element(By.XPATH, cashflow_count).click()
            webwait.until(invisibility(loading))
            sleep(4 * break_time)

            # click download excel
            dl_btn = "//*[@id='new-cash-flow-grid']/div/div[1]/span[2]/span"
            webwait.until(clickable((By.XPATH, dl_btn)))
            driver.find_element(By.XPATH, dl_btn).click()
            sleep(2 * break_time)

            # replace last file
            new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['cash'][time_type]}"
            move_last_file(new_path)
            sleep(2 * break_time)

            # open and ctrl + s excel file
            resave_excel(new_path)
            sleep(2 * break_time)

    except Exception as err:
        print(f"cant download cashflow {name} : {err}")


def bourseview_product_revenue(
    name,
    y=5,
    q=5,
    m=50,
    time_types=["yearly", "quarterly", "monthly"],
    report_types=["_seprated", ""],
):
    """download 6 files : (yearly,quarterly,monthly) (_seprated)
    <y = year : 5,10,20,50>
    <q = quarterly : 5,10,20,50>
    <m = month : 5,10,20,50>"""

    try:
        # select 'tolid va frosh'
        product = "//*[@id='stocks-sub-menu']/ul/li[8]/a[2]"
        webwait.until(clickable((By.XPATH, product)))
        driver.find_element(By.XPATH, product).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "/html").click()
        loading_xpath = "//*[@id='grid']/div/div[4]"
        loading = driver.find_element(By.XPATH, loading_xpath)
        webwait.until(invisibility(loading))
        sleep(4 * break_time)

        for report_type in report_types:
            # click 'tafkik dakheli,khareji'
            seprate = "//*[@id='grid']/div/div[1]/span[1]/div[3]"
            webwait.until(clickable((By.XPATH, seprate)))
            driver.find_element(By.XPATH, seprate).click()
            webwait.until(invisibility(loading))
            sleep(break_time)

            # download base excels
            for time_type in time_types:
                if time_type == "yearly":
                    n = y

                if time_type == "quarterly":
                    n = q

                if time_type == "monthly":
                    n = m

                # select 'nooe'
                product_type = f"//option[@value='{time_type}']"
                webwait.until(clickable((By.XPATH, product_type)))
                driver.find_element(By.XPATH, product_type).click()
                sleep(break_time)

                # select 'dore'
                product_count = f"//option[@value='{n}']"
                webwait.until(clickable((By.XPATH, product_count)))
                driver.find_element(By.XPATH, product_count).click()
                webwait.until(invisibility(loading))
                sleep(4 * break_time)

                # click download excel
                dl_btn = "//*[@id='grid']/div/div[1]/span[2]/span"
                webwait.until(clickable((By.XPATH, dl_btn)))
                driver.find_element(By.XPATH, dl_btn).click()
                sleep(2 * break_time)

                # replace last file
                new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['product'][time_type][report_type]}"
                move_last_file(new_path)
                sleep(2 * break_time)

                # open and ctrl + s excel file
                resave_excel(new_path)
                sleep(2 * break_time)

    except Exception as err:
        print(f"cant download product {name} : {err}")


def bourseview_cost(name, y=5, q=5, time_types=["yearly", "quarterly"]):
    """create 2 excel : yearly,quarterly
    <y = year : 5,10,20,50>
    <q = quarterly : 5,10,20,50>"""

    try:
        # select 'bahaye tamam shode'
        cost = "//*[@id='stocks-sub-menu']/ul/li[13]/a[2]"
        webwait.until(clickable((By.XPATH, cost)))
        driver.find_element(By.XPATH, cost).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "/html").click()
        loading_xpath = "//*[@id='grid-cogs']/div/div[1]"
        loading = driver.find_element(By.XPATH, loading_xpath)
        webwait.until(invisibility(loading))
        sleep(4 * break_time)

        # download excel
        for time_type in time_types:
            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            cost_type = f"//option[@value='{time_type}']"
            webwait.until(clickable((By.XPATH, cost_type)))
            driver.find_element(By.XPATH, cost_type).click()
            sleep(break_time)

            # select 'dore'
            cost_count = f"//option[@value='{n}']"
            webwait.until(clickable((By.XPATH, cost_count)))
            driver.find_element(By.XPATH, cost_count).click()
            webwait.until(invisibility(loading))
            sleep(4 * break_time)

            # click 'hameye' data
            all_data = "//*[@id='grid-cogs']/div/div[4]/div/div[1]"
            webwait.until(clickable((By.XPATH, all_data)))
            driver.find_element(By.XPATH, all_data).click()
            sleep(break_time)

            # go top of page
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.CONTROL + Keys.HOME)
            sleep(break_time)

            # click download excel
            dl_btn = "//*[@id='grid-cogs']/div/div[2]/span[2]/span"
            webwait.until(clickable((By.XPATH, dl_btn)))
            driver.find_element(By.XPATH, dl_btn).click()
            sleep(2 * break_time)

            # replace last file
            new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['cost'][time_type]}"
            move_last_file(new_path)
            sleep(2 * break_time)

            # open and ctrl + s excel file
            resave_excel(new_path)
            sleep(2 * break_time)

    except Exception as err:
        print(f"cant download cost {name} : {err}")


def bourseview_official(name, y=5, q=5, time_types=["yearly", "quarterly"]):
    """create 2 excel : yearly,quarterly
    <y = year : 5,10,20,50>
    <q = quarterly : 5,10,20,50>"""

    try:
        # select 'hazine haye omomi edari'
        official = "//*[@id='stocks-sub-menu']/ul/li[16]/a[2]"
        webwait.until(clickable((By.XPATH, official)))
        driver.find_element(By.XPATH, official).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "/html").click()
        loading_xpath = "//*[@id='grid']/div/div[1]"
        loading = driver.find_element(By.XPATH, loading_xpath)
        webwait.until(invisibility(loading))
        sleep(4 * break_time)

        # 'salane','fasli'
        for time_type in time_types:
            if time_type == "yearly":
                n = y

            if time_type == "quarterly":
                n = q

            # select 'nooe'
            official_type = f"//option[@value='{time_type}']"
            webwait.until(clickable((By.XPATH, official_type)))
            driver.find_element(By.XPATH, official_type).click()
            sleep(break_time)

            # select 'dore'
            official_count = f"//option[@value='{n}']"
            webwait.until(clickable((By.XPATH, official_count)))
            driver.find_element(By.XPATH, official_count).click()
            webwait.until(invisibility(loading))
            sleep(4 * break_time)

            # click download excel
            dl_btn = "//*[@id='grid']/div/div[2]/span[2]/span"
            webwait.until(clickable((By.XPATH, dl_btn)))
            driver.find_element(By.XPATH, dl_btn).click()
            sleep(2 * break_time)

            # replace last file
            new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['official'][time_type]}"
            move_last_file(new_path)
            sleep(2 * break_time)
            resave_excel(new_path)
            sleep(2 * break_time)

    except Exception as err:
        print(f"cant download official {name} : {err}")


def bourseview_price_history(name, start=year_ago, end=today_10char):
    """download pe
    <start : 1390/01/01>
    <end : 1400/01/01>"""
    try:
        # select 'tarikhche gheimat'
        price = "//*[@id='stocks-sub-menu']/ul/li[21]/a[2]"
        webwait.until(clickable((By.XPATH, price)))
        driver.find_element(By.XPATH, price).click()
        sleep(2 * break_time)

        # wait for loading page
        driver.find_element(By.XPATH, "/html").click()
        loading_xpath = "//*[@id='records-ag-grid']/div/div[2]/div[2]/div[6]/div/div"
        loading = driver.find_element(By.XPATH, loading_xpath)
        webwait.until(invisibility(loading))
        sleep(4 * break_time)

        # send 'tarikh shoroea'
        first = "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/input"
        webwait.until(presence((By.XPATH, first)))
        driver.find_element(By.XPATH, first).clear()
        driver.find_element(By.XPATH, first).send_keys(start)
        sleep(break_time)

        # send 'tarikh payan'
        last = "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/div[3]/div[1]/input"
        webwait.until(presence((By.XPATH, last)))
        driver.find_element(By.XPATH, last).clear()
        driver.find_element(By.XPATH, last).send_keys(end)
        sleep(break_time)

        # click 'namayesh gheimat'
        show = "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[1]/button"
        webwait.until(clickable((By.XPATH, show)))
        driver.find_element(By.XPATH, show).click()
        sleep(break_time)

        # click download excel
        dl_btn = "//*[@id='stocks-content-body']/div[1]/div[2]/div[1]/div[2]/div/div/div[2]/div"
        webwait.until(clickable((By.XPATH, dl_btn)))
        driver.find_element(By.XPATH, dl_btn).click()
        sleep(2 * break_time)

        # replace last file
        new_path = f"{INDUSPATH}/{wl_prod[name]['indus']}/{name}/{structure['pe']}"
        move_last_file(new_path)
        sleep(2 * break_time)

        # open and ctrl + s excel file
        resave_excel(new_path)
        sleep(2 * break_time)

    except Exception as err:
        print(f"cant download price history of {name} : {err}")


def bourseview_macro(start=year_ago, end=today_10char):
    """download macro
    <start : 1390/01/01>
    <end : 1400/01/01>"""
    try:
        # select 'dadehaye kalan'
        macro = "//*[@id='step3']/li[9]"
        webwait.until(clickable((By.XPATH, macro)))
        driver.find_element(By.XPATH, macro).click()
        sleep(2 * break_time)

        # select 'shakhes boors iran'
        index = "//*[@id='macro-history-select-irex-wrapper']/span"
        webwait.until(clickable((By.XPATH, index)))
        driver.find_element(By.XPATH, index).click()
        sleep(break_time)

        # click 'arzesh moamelat'
        value = "/html/body/span/span/span/ul/li/ul/li[3]"
        webwait.until(clickable((By.XPATH, value)))
        driver.find_element(By.XPATH, value).click()
        sleep(break_time)

        # click p/e ttm
        p_e = "/html/body/span/span/span/ul/li/ul/li[6]"
        webwait.until(clickable((By.XPATH, p_e)))
        driver.find_element(By.XPATH, p_e).click()
        sleep(break_time)

        # click p/d
        p_d = "/html/body/span/span/span/ul/li/ul/li[8]"
        webwait.until(clickable((By.XPATH, p_d)))
        driver.find_element(By.XPATH, p_d).click()
        sleep(break_time)

        # select 'saham shakhes boors'
        stocks = "//*[@id='macro-history-select-tedpix-wrapper']/span"
        webwait.until(clickable((By.XPATH, stocks)))
        driver.find_element(By.XPATH, stocks).click()
        sleep(break_time)

        # click 'arzesh khales vorod haghighi'
        real = "/html/body/span/span/span/ul/li/ul/li[20]"
        webwait.until(clickable((By.XPATH, real)))
        driver.find_element(By.XPATH, real).click()
        sleep(break_time)

        # select 'nerkh arz'
        currency = "//*[@id='macro-history-select-currency-wrapper']/span"
        webwait.until(clickable((By.XPATH, currency)))
        driver.find_element(By.XPATH, currency).click()
        sleep(break_time)

        # click dollar 'nima'
        dollar = "/html/body/span/span/span/ul/li/ul/li[1]"
        webwait.until(clickable((By.XPATH, dollar)))
        driver.find_element(By.XPATH, dollar).click()
        sleep(break_time)

        # click dollar 'azad'
        free = "/html/body/span/span/span/ul/li/ul/li[2]"
        webwait.until(clickable((By.XPATH, free)))
        driver.find_element(By.XPATH, free).click()
        sleep(break_time)

        # select 'motoghayerhaye poli'
        money = "//*[@id='macro-history-select-economics-wrapper']/span"
        webwait.until(clickable((By.XPATH, money)))
        driver.find_element(By.XPATH, money).click()
        sleep(break_time)

        # click 'miangin nerkh bedon risk'
        risk = "/html/body/span/span/span/ul/li/ul/li[2]"
        webwait.until(clickable((By.XPATH, risk)))
        driver.find_element(By.XPATH, risk).click()
        sleep(break_time)

        # click download button
        dl_exl = "/html/body/div[2]/div/div/div/div/div[3]/div[3]/div/div[1]/div/div/div[3]/div/span"
        webwait.until(clickable((By.XPATH, dl_exl)))
        driver.find_element(By.XPATH, dl_exl).click()
        sleep(break_time)

        # click 'entekhab baze delkhah'
        limit = "//*[@id='myModal']/div/div/div[2]/div/button"
        webwait.until(clickable((By.XPATH, limit)))
        driver.find_element(By.XPATH, limit).click()
        sleep(break_time)

        # send 'tarikh shoroe'
        first = "//*[@id='collapseExample']/div/div/div[1]/div[1]/input"
        webwait.until(presence((By.XPATH, first)))
        driver.find_element(By.XPATH, first).send_keys(start)
        sleep(break_time)

        # send 'tarikh payan'
        last = "//*[@id='collapseExample']/div/div/div[3]/div[1]/input"
        webwait.until(presence((By.XPATH, last)))
        driver.find_element(By.XPATH, last).send_keys(end)
        sleep(4 * break_time)

        # click download excel
        dl_btn = "//*[@id='myModal']/div/div/div[3]/button"
        webwait.until(clickable((By.XPATH, dl_btn)))
        driver.find_element(By.XPATH, dl_btn).click()
        sleep(2 * break_time)

        # click blank page
        driver.find_element(By.XPATH, "/html").click()
        sleep(break_time)

        # replace last file
        new_path = f"{MACROPATH}/macro.xlsx"
        move_last_file(new_path)
        sleep(2 * break_time)

    except Exception as err:
        print(f"cant download macro data : {err}")


def integrate_database(stocks=wl_prod_keys, y=5, q=10, m=50):
    """download deficiencies of stock files"""
    create_database_structure()

    bourseview_login()
    for name in stocks:
        print(f"integrate {name} ...")
        deficiencies = find_deficiencies(name)[0]
        if deficiencies != {}:
            bourseview_search(name)
            for d in deficiencies:
                if d == "balancesheet":
                    bourseview_balancesheet(name, time_types=deficiencies[d], y=y, q=q)
                if d == "income":
                    bourseview_income_statement(
                        name, time_types=deficiencies[d], y=y, q=q
                    )
                if d == "product":
                    bourseview_product_revenue(
                        name, time_types=deficiencies[d], y=y, q=q, m=m
                    )
                if d == "official":
                    bourseview_official(name, time_types=deficiencies[d], y=y, q=q)
                if d == "cashflow":
                    bourseview_cashflow(name, time_types=deficiencies[d], y=y, q=q)
                if d == "cost":
                    bourseview_cost(name, time_types=deficiencies[d], y=y, q=q)

        if find_deficiencies(name)[1]:
            bourseview_search(name)
            bourseview_price_history(name)

    codal_login()
    for name in stocks:
        if find_deficiencies(name)[3]:
            codal_search(name)
            codal_eps(name)


def update_database(
    stocks=wl_prod_keys,
    yearly=False,
    quarterly=False,
    monthly=False,
    dl_pe=True,
    dl_eps=True,
    y=5,
    q=10,
    m=50,
):
    """
    update 22 excel of stock : 8 yearly + 10 quarterly + 2 monthly + pe + eps
    """
    create_database_structure()

    t = []
    if yearly:
        t.append("yearly")

    if quarterly:
        t.append("quarterly")

    t2 = t.copy()
    if monthly:
        t2.append("monthly")

    bourseview_login()
    for name in stocks:
        print(f"update {name} ...")
        bourseview_search(name)
        bourseview_balancesheet(name, time_types=t, y=y, q=q)
        bourseview_income_statement(name, time_types=t, y=y, q=q)
        bourseview_cashflow(name, time_types=t, y=y, q=q)
        bourseview_product_revenue(name, time_types=t2, y=y, q=q, m=m)
        bourseview_cost(name, time_types=t, y=y, q=q)
        bourseview_official(name, time_types=t, y=y, q=q)
        if dl_pe:
            bourseview_price_history(name)

    bourseview_macro()

    if yearly and dl_eps:
        codal_login()
        for name in stocks:
            codal_search(name)
            codal_eps(name)


if __name__ == "__main__":
    integrate_database()
    for i in wl_prod:
        check_stock_files(i, action=True)
        print(i, find_deficiencies(i))
