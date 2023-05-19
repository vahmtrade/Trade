import os
import pandas as pd
from persiantools.jdatetime import JalaliDate

wl_prod = {
    "fakhouz": {
        "indus": "folad",
        "token": "فخوز",
        "name": "فخوز",
        "scenario": "dollar",
    },
    "folad": {
        "indus": "folad",
        "token": "فولاد",
        "name": "فولاد مبارکه اصفهان",
        "scenario": "dollar",
    },
    "sekhouz": {
        "indus": "siman",
        "token": "سخوز",
        "name": "سیمان خوزستان",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "shekarbon": {
        "indus": "dode",
        "token": "شکربن",
        "name": "کربن ایران",
        "scenario": "dollar",
    },
    "kermasha": {
        "indus": "urea",
        "token": "کرماشا",
        "name": "صنایع پتروشیمی کرمانشاه",
        "scenario": "dollar",
    },
    "khorasan": {
        "indus": "urea",
        "token": "خراسان",
        "name": "پتروشیمی خراسان",
        "scenario": "dollar",
    },
    "shapdis": {
        "indus": "urea",
        "token": "شپدیس",
        "name": "پتروشیمی پردیس",
        "scenario": "dollar",
    },
    "pekavir": {
        "indus": "lastic",
        "token": "پکویر",
        "name": "کویر تایر",
        "scenario": "last",
        "alpha_rate": 1.25,
        "alpha_rate_next": 1.4,
    },
    "pekerman": {
        "indus": "lastic",
        "token": "پکرمان",
        "name": "گروه صنعتی بارز",
        "scenario": "last",
        "alpha_rate": 1.25,
        "alpha_rate_next": 1.4,
    },
    "sekhash": {
        "indus": "siman",
        "token": "سخاش",
        "name": "سیمان خاش",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "semazen": {
        "indus": "siman",
        "token": "سمازن",
        "name": "سیمان مازندران",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "saroum": {
        "indus": "siman",
        "token": "ساروم",
        "name": "سیمان ارومیه",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "sekord": {
        "indus": "siman",
        "token": "سکرد",
        "name": "سیمان کردستان",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "sesoufi": {
        "indus": "siman",
        "token": "سصوفی",
        "name": "سیمان صوفیان",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "seshargh": {
        "indus": "siman",
        "token": "سشرق",
        "name": "سشرق",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "pasa": {
        "indus": "lastic",
        "token": "پاسا",
        "name": "ایران یاسا تایر و رابر",
        "scenario": "last",
        "alpha_rate": 1.25,
        "alpha_rate_next": 1.4,
    },
    "petayer": {
        "indus": "lastic",
        "token": "پتایر",
        "name": "ایران تایر",
        "scenario": "last",
        "alpha_rate": 1.25,
        "alpha_rate_next": 1.4,
    },
    "zagros": {
        "indus": "methanol",
        "token": "زاگرس",
        "name": "پتروشیمی زاگرس",
        "scenario": "dollar",
    },
    "shefan": {
        "indus": "methanol",
        "token": "شفن",
        "name": "پتروشیمی فن آوران",
        "scenario": "dollar",
    },
    "shekhark": {
        "indus": "methanol",
        "token": "شخارک",
        "name": "پتروشیمی خارک",
        "scenario": "dollar",
    },
    "parta": {
        "indus": "lastic",
        "token": "پارتا",
        "name": "مجتمع صنعتی ارتا ویل تایر",
        "scenario": "last",
        "alpha_rate": 1.25,
        "alpha_rate_next": 1.4,
    },
    "shapna": {
        "indus": "palayesh",
        "token": "شپنا",
        "name": "پالایش نفت اصفهان",
        "scenario": "dollar",
    },
    "shesadaf": {
        "indus": "dode",
        "token": "شصدف",
        "name": "صنعتی دوده فام",
        "scenario": "dollar",
    },
    "shedoos": {
        "indus": "dode",
        "token": "شدوص",
        "name": "دوده صنعتی پارس",
        "scenario": "dollar",
    },
    "sehegmat": {
        "indus": "siman",
        "token": "سهگمت",
        "name": "سیمان هگمتان",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "ghegolpa": {
        "indus": "ghaza",
        "token": "غگلپا",
        "name": "پگاه گلپایگان",
        "scenario": "last",
        "alpha_rate": 1.2,
        "alpha_rate_next": 1.2,
    },
    "gheshasfa": {
        "indus": "ghaza",
        "token": "غشصفا",
        "name": "پگاه اصفهان",
        "scenario": "last",
        "alpha_rate": 1.2,
        "alpha_rate_next": 1.2,
    },
    "ghekurosh": {
        "indus": "ghaza",
        "token": "غکورش",
        "name": "صنعت غذایی کورش",
        "scenario": "last",
        "alpha_rate": 1.2,
        "alpha_rate_next": 1.2,
    },
    "ghegorji": {
        "indus": "ghaza",
        "token": "غگرجی",
        "name": "بیسکویت گرجی",
        "scenario": "last",
        "alpha_rate": 1.35,
        "alpha_rate_next": 1.6,
    },
    "faspa": {
        "indus": "folad",
        "token": "فسپا",
        "name": "گروه صنعتی سپاهان",
        "scenario": "dollar",
    },
    "kesave": {
        "indus": "kashi",
        "token": "کساوه",
        "name": "کاشی و سرامیک سینا",
        "scenario": "last",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4,
    },
    "detmad": {
        "indus": "darou",
        "token": "دتماد",
        "name": "تولید مواد اولیه داروپخش",
        "scenario": "last",
        "alpha_rate": 1.3,
        "alpha_rate_next": 1.5,
    },
    "ghezar": {
        "indus": "ghaza",
        "token": "غزر",
        "name": "زر ماکارون",
        "scenario": "last",
        "alpha_rate": 1.3,
        "alpha_rate_next": 1.5,
    },
    "delor": {
        "indus": "darou",
        "token": "دلر",
        "name": "داروسازی اکسیر",
        "scenario": "last",
        "alpha_rate": 1.3,
        "alpha_rate_next": 1.5,
    },
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
    "ofogh": {
        "indus": "foroushgah",
        "token": "افق",
        "name": "فروشگاه های زنجیره ای افق",
    },
    "kave": {"indus": "folad", "token": "کاوه", "name": "فولاد کاوه جنوب کیش"},
    "hormoz": {"indus": "folad", "token": "هرمز", "name": "فولاد هرمزگان جنوب"},
    "ghebshahr": {"indus": "ghaza", "token": "غبشهر", "name": "صنعتی بهشهر"},
    "sharum": {"indus": "chemical", "token": "شاروم", "name": "پتروشیمی ارومیه"},
    "jam": {"indus": "chemical", "token": "جم", "name": "پترو شیمی جم"},
    "derazak": {"indus": "darou", "token": "درازک", "name": "لابراتوارهای رازک"},
    "kavir": {"indus": "folad", "token": "کویر", "name": "تولیدی فولاد سپید فراب کویر"},
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
    "value_opt": "value_opt.xlsx",
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


def find_project_root(project_name):
    current_path = os.path.abspath(".")
    while True:
        project_path = os.path.join(current_path, project_name)
        if os.path.isdir(project_path):
            return project_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            return None
        current_path = parent_path


ROOT_PATH = find_project_root("Trade").replace("\\", "/")
DB = f"{ROOT_PATH}/database"
WINDOWS_FIREFOX_DRIVER_PATH = f"{ROOT_PATH}/statics/geckodriver.exe"
LINUX_FIREFOX_DRIVER_PATH = f"{ROOT_PATH}/statics/geckodriver"

INDUSPATH = f"{DB}/industries"
MACROPATH = f"{DB}/macro"
FOREXPATH = f"{DB}/forex"
PKLPATH = f"{DB}/watchlist"
