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


def seprate_number(a, seprator=","):
    """12345678 => 12,345,678"""
    try:
        a = str(int(float(a)))
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

    return b
