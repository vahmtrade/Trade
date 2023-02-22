import os
import pandas as pd
from persiantools.jdatetime import JalaliDate

wl_prod = {
    "fakhouz": {"indus": "folad", "token": "فخوز", "name": "فخوز"},
    "folad": {"indus": "folad", "token": "فولاد", "name": "فولاد مبارکه اصفهان"},
    "sekhouz": {"indus": "siman", "token": "سخوز", "name": "سیمان خوزستان"},
    "shekarbon": {"indus": "dode", "token": "شکربن", "name": "کربن ایران"},
    "kermasha": {"indus": "urea", "token": "کرماشا", "name": "صنایع پتروشیمی کرمانشاه"},
    "khorasan": {"indus": "urea", "token": "خراسان", "name": "پتروشیمی خراسان"},
    "shapdis": {"indus": "urea", "token": "شپدیس", "name": "پتروشیمی پردیس"},
    "pekavir": {"indus": "lastic", "token": "پکویر", "name": "کویر تایر"},
    "pekerman": {"indus": "lastic", "token": "پکرمان", "name": "گروه صنعتی بارز"},
    "sekhash": {"indus": "siman", "token": "سخاش", "name": "سیمان خاش"},
    "semazen": {"indus": "siman", "token": "سمازن", "name": "سیمان مازندران"},
    "saroum": {"indus": "siman", "token": "ساروم", "name": "سیمان ارومیه"},
    "sekord": {"indus": "siman", "token": "سکرد", "name": "سیمان کردستان"},
    "sesoufi": {"indus": "siman", "token": "سصوفی", "name": "سیمان صوفیان"},
    "seshargh": {"indus": "siman", "token": "سشرق", "name": "سشرق"},
    "pasa": {"indus": "lastic", "token": "پاسا", "name": "ایران یاسا تایر و رابر"},
    "petayer": {"indus": "lastic", "token": "پتایر", "name": "ایران تایر"},
    "zagros": {"indus": "methanol", "token": "زاگرس", "name": "پتروشیمی زاگرس"},
    "shefan": {"indus": "methanol", "token": "شفن", "name": "پتروشیمی فن آوران"},
    "shekhark": {"indus": "methanol", "token": "شخارک", "name": "پتروشیمی خارک"},
    "parta": {"indus": "lastic", "token": "پارتا", "name": "مجتمع صنعتی ارتا ویل تایر"},
    "shapna": {"indus": "palayesh", "token": "شپنا", "name": "پالایش نفت اصفهان"},
    "shesadaf": {"indus": "dode", "token": "شصدف", "name": "صنعتی دوده فام"},
    "shedoos": {"indus": "dode", "token": "شدوص", "name": "دوده صنعتی پارس"},
    "sehegmat": {"indus": "siman", "token": "سهگمت", "name": "سیمان هگمتان"},
    "ghegolpa": {"indus": "ghaza", "token": "غگلپا", "name": "پگاه گلپایگان"},
    "gheshasfa": {"indus": "ghaza", "token": "غشصفا", "name": "پگاه اصفهان"},
    "ghekurosh": {"indus": "ghaza", "token": "غکورش", "name": "صنعت غذایی کورش"},
    "ghegorji": {"indus": "ghaza", "token": "غگرجی", "name": "بیسکویت گرجی"},
    "faspa": {"indus": "folad", "token": "فسپا", "name": "گروه صنعتی سپاهان"},
    "kesave": {"indus": "kashi", "token": "کساوه", "name": "کاشی و سرامیک سینا"},
    "detmad": {"indus": "darou", "token": "دتماد", "name": "تولید مواد اولیه داروپخش"},
    "ghezar": {"indus": "ghaza", "token": "غزر", "name": "زر ماکارون"},
    "delor": {"indus": "darou", "token": "دلر", "name": "داروسازی اکسیر"},
    "kehamda": {"indus": "shishe", "token": "کهمدا", "name": "شیشه همدان"},
    "kehafez": {"indus": "kashi", "token": "کحافظ", "name": "کاشی و سرامیک حافظ"},
    "shaspa": {"indus": "ravankar", "token": "شسپا", "name": "نفت سپاهان"},
    "shepaksa": {"indus": "shoyande", "token": "شپاکسا", "name": "پاکسان"},
    "ghefars": {"indus": "ghaza", "token": "غفارس", "name": "پگاه فارس"},
    "bemoto": {"indus": "bargh", "token": "بموتو", "name": "موتوژن"},
    "gheshasfa": {"indus": "ghaza", "token": "غشصفا", "name": "پگاه اصفهان"},
    "desobhan": {"indus": "darou", "token": "دسبحان", "name": "سبحان دارو"},
    "deshimi": {"indus": "darou", "token": "دشیمی", "name": "شیمی دارویی داروپخش"},
    "khedizel": {"indus": "khodro", "token": "خدیزل", "name": "بهمن دیزل"},
    "fameli": {"indus": "felezat", "token": "فملی", "name": "فملی "},
    "kechad": {"indus": "folad", "token": "کچاد", "name": "معدنی و صنعتی چادرملو"},
    "fasmin": {"indus": "felezat", "token": "فاسمین", "name": "کالسیمین"},
    "shebandar": {"indus": "palayesh", "token": "شبندر", "name": "پالایش نفت بندر"},
    "shetran": {"indus": "palayesh", "token": "شتران", "name": "پالایش نفت تهران"},
    "gheshan": {"indus": "ghaza", "token": "غشان", "name": "پگاه خراسان"},
    "deabid": {"indus": "darou", "token": "دعبید", "name": "دکتر عبیدی"},
    "dekapsul": {"indus": "darou", "token": "دکپسول", "name": "کپسول ایران"},
    "gharn": {"indus": "shoyande", "token": "قرن", "name": "پدیده شیمی قرن"},
    "shekolor": {"indus": "methanol", "token": "شکلر", "name": "نیروکلر"},
    "shegol": {"indus": "shoyande", "token": "شگل", "name": "گلتاش"},
    "silam": {"indus": "siman", "token": "سیلام", "name": "سیمان ایلام"},
    "ghesafha": {"indus": "ghand", "token": "قصفها", "name": "قند اصفهان"},
    "shavan": {"indus": "palayesh", "token": "شاوان", "name": "پالایش نفت لاوان"},
    "defara": {"indus": "darou", "token": "دفارا", "name": "داروسازی فارابی"},
    "dalber": {"indus": "darou", "token": "دالبر", "name": "البرز دارو"},
    "kimia": {
        "indus": "felezat",
        "token": "کیمیا",
        "name": "معدنی کیمیای زنجان گستران",
    },
    "gheshahdab": {"indus": "ghaza", "token": "غشهداب", "name": "کشت و صنعت شهداب"},
    "ghepino": {"indus": "ghaza", "token": "غپینو", "name": "پارس مینو"},
    "simorgh": {"indus": "zeraat", "token": "سیمرغ", "name": "سیمرغ"},
    "save": {"indus": "siman", "token": "ساوه", "name": "سیمان ساوه"},
    "shamla": {"indus": "chemical", "token": "شاملا", "name": "معدنی املاح ایران"},
    "dejaber": {"indus": "darou", "token": "دجابر", "name": "داروسازی جابر ابن حیان"},
    "feghadir": {
        "indus": "folad",
        "token": "فغدیر",
        "name": "آهن و فولاد غدیر ایرانیان",
    },
    "kimiatec": {"indus": "shoyande", "token": "کیمیاتک", "name": "آریان کیمیا تک"},
    "ghesalem": {"indus": "ghaza", "token": "غسالم", "name": "سالمین"},
    "ghevita": {"indus": "ghaza", "token": "غویتا", "name": "ویتانا"},
    "faira": {"indus": "felezat", "token": "فایرا", "name": "آلومینیوم‌ایران"},
    "khetrak": {
        "indus": "khodro",
        "token": "ختراک",
        "name": "ریخته گری تراکتور سازی ایران",
    },
    "chekapa": {"indus": "kaghaz", "token": "چکاپا", "name": "گروه صنایع کاغذ پارس"},
    "zegoldasht": {
        "indus": "zeraat",
        "token": "زگلدشت",
        "name": "کشت و دام گلدشت نمونه اصفهان",
    },
    "kazar": {"indus": "kashi", "token": "کاذر", "name": "فرآورده‌های‌ نسوزآذر"},
    "netrin": {"indus": "nasaji", "token": "نطرین", "name": "عطرین نخ قم"},
    "fejahan": {"indus": "folad", "token": "فجهان", "name": "مجتمع جهان فولاد سیرجان"},
    "khodro": {"indus": "khodro", "token": "خودرو", "name": "ایران خودرو"},
    "kemina": {"indus": "shishe", "token": "کمینا", "name": "شیشه سازی مینا"},
    "sheranol": {"indus": "ravankar", "token": "شرانل", "name": "نفت ایرانول"},
}
wl_nprod = {
    "kian": {"indus": "fixed_income", "token": "کیان", "name": "کیان"},
    "palayesh": {"indus": "etf", "token": "پالایش", "name": "پالایش"},
    "fars": {"indus": "holding", "token": "فارس", "name": "صنایع پتروشیمی خلیج فارس"},
    "vapoya": {"indus": "holding", "token": "وپویا", "name": "سرمایه گذاری پویا"},
    "darayekom": {"indus": "etf", "token": "دارا یکم", "name": "دارا یکم"},
    "tala": {"indus": "tala", "token": "طلا", "name": "طلا"},
    "madira": {"indus": "domestic", "token": "مادیرا", "name": "صنایع مادیران"},
    "shasta": {
        "indus": "holding",
        "token": "شستا",
        "name": "سرمایه گذاری تأمین اجتماعی",
    },
}
structure = {
    "balance": {
        "yearly": "balancesheet/yearly.xlsx",
        "quarterly": "balancesheet/quarterly.xlsx",
    },
    "income": {
        "yearly": {
            "": {
                "rial": "income/yearly/rial.xlsx",
                "dollar": "income/yearly/dollar.xlsx",
            }
        },
        "quarterly": {
            "": {
                "rial": "income/quarterly/rial.xlsx",
                "dollar": "income/quarterly/dollar.xlsx",
            },
            "_cumulative": {
                "rial": "income/quarterly/rial_cumulative.xlsx",
                "dollar": "income/quarterly/dollar_cumulative.xlsx",
            },
        },
    },
    "product": {
        "yearly": {
            "": "product/yearly.xlsx",
            "_seprated": "product/yearly_seprated.xlsx",
        },
        "quarterly": {
            "": "product/quarterly.xlsx",
            "_seprated": "product/quarterly_seprated.xlsx",
        },
        "monthly": {
            "": "product/monthly.xlsx",
            "_seprated": "product/monthly_seprated.xlsx",
        },
    },
    "official": {
        "yearly": "official/yearly.xlsx",
        "quarterly": "official/quarterly.xlsx",
    },
    "cash": {"yearly": "cashflow/yearly.xlsx", "quarterly": "cashflow/quarterly.xlsx"},
    "cost": {"yearly": "cost/yearly.xlsx", "quarterly": "cost/quarterly.xlsx"},
    "analyse": "analyse/",
    "detail": "detail/",
    "pe": "pe/pe.xlsx",
    "eps": "eps.xlsx",
    "opt": "opt.xlsx",
}

wl_prod_df = pd.DataFrame(wl_prod).T
wl_prod_keys = list(wl_prod.keys())

today = JalaliDate.today()
today_8char = f"{today.year:02d}{today.month:02d}{today.day:02d}"
today_10char = f"{today.year:02d}/{today.month:02d}/{today.day:02d}"
year_ago = f"{today.year-10:02d}/{today.month:02d}/{today.day:02d}"
month_ago = f"{today.year:02d}/{today.month-1:02d}/{today.day:02d}"

regex_per_timeid_y = "[۰۱۲۳۴۵۶۷۸۹]{4}/[۰۱۲۳۴۵۶۷۸۹]{2}/[۰۱۲۳۴۵۶۷۸۹]{2}"
regex_en_timeid_q = "[0123456789]{4}/[0123456789]{2}"

ROOT_PATH = os.path.abspath(os.curdir).replace("\\", "/")
DB = f"{ROOT_PATH}/database"
WINDOWS_FIREFOX_DRIVER_PATH = f"{ROOT_PATH}/statics/geckodriver.exe"
LINUX_FIREFOX_DRIVER_PATH = f"{ROOT_PATH}/statics/geckodriver"

INDUSPATH = f"{DB}/industries"
MACROPATH = f"{DB}/macro"
FOREXPATH = f"{DB}/forex"
PKLPATH = f"{DB}/watchlist"
