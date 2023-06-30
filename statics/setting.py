import os
import pandas as pd
from persiantools.jdatetime import JalaliDate

wl_prod = {
    "fakhouz": {"indus": "folad", "token": "فخوز", "name": "فخوز", "fiscal_year": 12},
    "folad": {
        "indus": "folad",
        "token": "فولاد",
        "name": "فولاد مبارکه اصفهان",
        "fiscal_year": 12,
    },
    "sekhouz": {
        "indus": "siman",
        "token": "سخوز",
        "name": "سیمان خوزستان",
        "fiscal_year": 12,
    },
    "shekarbon": {
        "indus": "dode",
        "token": "شکربن",
        "name": "کربن ایران",
        "fiscal_year": 12,
    },
    "kermasha": {
        "indus": "urea",
        "token": "کرماشا",
        "name": "صنایع پتروشیمی کرمانشاه",
        "fiscal_year": 12,
    },
    "khorasan": {
        "indus": "urea",
        "token": "خراسان",
        "name": "پتروشیمی خراسان",
        "fiscal_year": 12,
    },
    "shapdis": {
        "indus": "urea",
        "token": "شپدیس",
        "name": "پتروشیمی پردیس",
        "fiscal_year": 6,
    },
    "pekavir": {
        "indus": "lastic",
        "token": "پکویر",
        "name": "کویر تایر",
        "fiscal_year": 9,
    },
    "pekerman": {
        "indus": "lastic",
        "token": "پکرمان",
        "name": "گروه صنعتی بارز",
        "fiscal_year": 12,
    },
    "sekhash": {
        "indus": "siman",
        "token": "سخاش",
        "name": "سیمان خاش",
        "fiscal_year": 12,
    },
    "semazen": {
        "indus": "siman",
        "token": "سمازن",
        "name": "سیمان مازندران",
        "fiscal_year": 9,
    },
    "saroum": {
        "indus": "siman",
        "token": "ساروم",
        "name": "سیمان ارومیه",
        "fiscal_year": 12,
    },
    "sekord": {
        "indus": "siman",
        "token": "سکرد",
        "name": "سیمان کردستان",
        "fiscal_year": 6,
    },
    "sesoufi": {
        "indus": "siman",
        "token": "سصوفی",
        "name": "سیمان صوفیان",
        "fiscal_year": 12,
    },
    "seshargh": {"indus": "siman", "token": "سشرق", "name": "سشرق", "fiscal_year": 6},
    "pasa": {
        "indus": "lastic",
        "token": "پاسا",
        "name": "ایران یاسا تایر و رابر",
        "fiscal_year": 12,
    },
    "petayer": {
        "indus": "lastic",
        "token": "پتایر",
        "name": "ایران تایر",
        "fiscal_year": 12,
    },
    "zagros": {
        "indus": "methanol",
        "token": "زاگرس",
        "name": "پتروشیمی زاگرس",
        "fiscal_year": 12,
    },
    "shefan": {
        "indus": "methanol",
        "token": "شفن",
        "name": "پتروشیمی فن آوران",
        "fiscal_year": 12,
    },
    "shekhark": {
        "indus": "methanol",
        "token": "شخارک",
        "name": "پتروشیمی خارک",
        "fiscal_year": 12,
    },
    "parta": {
        "indus": "lastic",
        "token": "پارتا",
        "name": "مجتمع صنعتی ارتا ویل تایر",
        "fiscal_year": 12,
    },
    "shapna": {
        "indus": "palayesh",
        "token": "شپنا",
        "name": "پالایش نفت اصفهان",
        "fiscal_year": 12,
    },
    "shesadaf": {
        "indus": "dode",
        "token": "شصدف",
        "name": "صنعتی دوده فام",
        "fiscal_year": 12,
    },
    "shedoos": {
        "indus": "dode",
        "token": "شدوص",
        "name": "دوده صنعتی پارس",
        "fiscal_year": 12,
    },
    "sehegmat": {
        "indus": "siman",
        "token": "سهگمت",
        "name": "سیمان هگمتان",
        "fiscal_year": 10,
    },
    "ghegolpa": {
        "indus": "ghaza",
        "token": "غگلپا",
        "name": "پگاه گلپایگان",
        "fiscal_year": 12,
    },
    "gheshasfa": {
        "indus": "ghaza",
        "token": "غشصفا",
        "name": "پگاه اصفهان",
        "fiscal_year": 12,
    },
    "ghekurosh": {
        "indus": "ghaza",
        "token": "غکورش",
        "name": "صنعت غذایی کورش",
        "fiscal_year": 12,
    },
    "ghegorji": {
        "indus": "ghaza",
        "token": "غگرجی",
        "name": "بیسکویت گرجی",
        "fiscal_year": 12,
    },
    "faspa": {
        "indus": "folad",
        "token": "فسپا",
        "name": "گروه صنعتی سپاهان",
        "fiscal_year": 12,
    },
    "kesave": {
        "indus": "kashi",
        "token": "کساوه",
        "name": "کاشی و سرامیک سینا",
        "fiscal_year": 12,
    },
    "detmad": {
        "indus": "darou",
        "token": "دتماد",
        "name": "تولید مواد اولیه داروپخش",
        "fiscal_year": 12,
    },
    "ghezar": {
        "indus": "ghaza",
        "token": "غزر",
        "name": "زر ماکارون",
        "fiscal_year": 12,
    },
    "delor": {
        "indus": "darou",
        "token": "دلر",
        "name": "داروسازی اکسیر",
        "fiscal_year": 12,
    },
    "kehamda": {
        "indus": "shishe",
        "token": "کهمدا",
        "name": "شیشه همدان",
        "fiscal_year": 3,
    },
    "kehafez": {
        "indus": "kashi",
        "token": "کحافظ",
        "name": "کاشی و سرامیک حافظ",
        "fiscal_year": 12,
    },
    "shaspa": {
        "indus": "ravankar",
        "token": "شسپا",
        "name": "نفت سپاهان",
        "fiscal_year": 12,
    },
    "shepaksa": {
        "indus": "shoyande",
        "token": "شپاکسا",
        "name": "پاکسان",
        "fiscal_year": 9,
    },
    "ghefars": {
        "indus": "ghaza",
        "token": "غفارس",
        "name": "پگاه فارس",
        "fiscal_year": 12,
    },
    "bemoto": {"indus": "bargh", "token": "بموتو", "name": "موتوژن", "fiscal_year": 6},
    "desobhan": {
        "indus": "darou",
        "token": "دسبحان",
        "name": "سبحان دارو",
        "fiscal_year": 12,
    },
    "deshimi": {
        "indus": "darou",
        "token": "دشیمی",
        "name": "شیمی دارویی داروپخش",
        "fiscal_year": 12,
    },
    "khedizel": {
        "indus": "khodro",
        "token": "خدیزل",
        "name": "بهمن دیزل",
        "fiscal_year": 12,
    },
    "fameli": {"indus": "felezat", "token": "فملی", "name": "فملی ", "fiscal_year": 12},
    "kechad": {
        "indus": "folad",
        "token": "کچاد",
        "name": "معدنی و صنعتی چادرملو",
        "fiscal_year": 12,
    },
    "fasmin": {
        "indus": "felezat",
        "token": "فاسمین",
        "name": "کالسیمین",
        "fiscal_year": 12,
    },
    "shebandar": {
        "indus": "palayesh",
        "token": "شبندر",
        "name": "پالایش نفت بندر",
        "fiscal_year": 12,
    },
    "shetran": {
        "indus": "palayesh",
        "token": "شتران",
        "name": "پالایش نفت تهران",
        "fiscal_year": 12,
    },
    "gheshan": {
        "indus": "ghaza",
        "token": "غشان",
        "name": "پگاه خراسان",
        "fiscal_year": 12,
    },
    "deabid": {
        "indus": "darou",
        "token": "دعبید",
        "name": "دکتر عبیدی",
        "fiscal_year": 9,
    },
    "dekapsul": {
        "indus": "darou",
        "token": "دکپسول",
        "name": "کپسول ایران",
        "fiscal_year": 12,
    },
    "gharn": {
        "indus": "shoyande",
        "token": "قرن",
        "name": "پدیده شیمی قرن",
        "fiscal_year": 12,
    },
    "shekolor": {
        "indus": "methanol",
        "token": "شکلر",
        "name": "نیروکلر",
        "fiscal_year": 8,
    },
    "shegol": {"indus": "shoyande", "token": "شگل", "name": "گلتاش", "fiscal_year": 9},
    "silam": {
        "indus": "siman",
        "token": "سیلام",
        "name": "سیمان ایلام",
        "fiscal_year": 10,
    },
    "ghesafha": {
        "indus": "ghand",
        "token": "قصفها",
        "name": "قند اصفهان",
        "fiscal_year": 12,
    },
    "shavan": {
        "indus": "palayesh",
        "token": "شاوان",
        "name": "پالایش نفت لاوان",
        "fiscal_year": 12,
    },
    "defara": {
        "indus": "darou",
        "token": "دفارا",
        "name": "داروسازی فارابی",
        "fiscal_year": 12,
    },
    "dalber": {
        "indus": "darou",
        "token": "دالبر",
        "name": "البرز دارو",
        "fiscal_year": 12,
    },
    "kimia": {
        "indus": "felezat",
        "token": "کیمیا",
        "name": "معدنی کیمیای زنجان گستران",
        "fiscal_year": 12,
    },
    "gheshahdab": {
        "indus": "ghaza",
        "token": "غشهداب",
        "name": "کشت و صنعت شهداب",
        "fiscal_year": 12,
    },
    "ghepino": {
        "indus": "ghaza",
        "token": "غپینو",
        "name": "پارس مینو",
        "fiscal_year": 12,
    },
    "simorgh": {
        "indus": "zeraat",
        "token": "سیمرغ",
        "name": "سیمرغ",
        "fiscal_year": 12,
    },
    "save": {
        "indus": "siman",
        "token": "ساوه",
        "name": "سیمان ساوه",
        "fiscal_year": 12,
    },
    "shamla": {
        "indus": "madani",
        "token": "شاملا",
        "name": "معدنی املاح ایران",
        "fiscal_year": 12,
    },
    "dejaber": {
        "indus": "darou",
        "token": "دجابر",
        "name": "داروسازی جابر ابن حیان",
        "fiscal_year": 12,
    },
    "feghadir": {
        "indus": "folad",
        "token": "فغدیر",
        "name": "آهن و فولاد غدیر ایرانیان",
        "fiscal_year": 12,
    },
    "kimiatec": {
        "indus": "shoyande",
        "token": "کیمیاتک",
        "name": "آریان کیمیا تک",
        "fiscal_year": 12,
    },
    "ghesalem": {
        "indus": "biscuit",
        "token": "غسالم",
        "name": "سالمین",
        "fiscal_year": 12,
    },
    "ghevita": {"indus": "ghaza", "token": "غویتا", "name": "ویتانا", "fiscal_year": 9},
    "faira": {
        "indus": "felezat",
        "token": "فایرا",
        "name": "آلومینیومایران",
        "fiscal_year": 12,
    },
    "khetrak": {
        "indus": "khodro",
        "token": "ختراک",
        "name": "ریخته گری تراکتور سازی ایران",
        "fiscal_year": 12,
    },
    "chekapa": {
        "indus": "kaghaz",
        "token": "چکاپا",
        "name": "گروه صنایع کاغذ پارس",
        "fiscal_year": 8,
    },
    "zegoldasht": {
        "indus": "zeraat",
        "token": "زگلدشت",
        "name": "کشت و دام گلدشت نمونه اصفهان",
        "fiscal_year": 12,
    },
    "kazar": {
        "indus": "kashi",
        "token": "کاذر",
        "name": "فرآوردههای نسوزآذر",
        "fiscal_year": 12,
    },
    "netrin": {
        "indus": "nasaji",
        "token": "نطرین",
        "name": "عطرین نخ قم",
        "fiscal_year": 12,
    },
    "fejahan": {
        "indus": "folad",
        "token": "فجهان",
        "name": "مجتمع جهان فولاد سیرجان",
        "fiscal_year": 12,
    },
    "khodro": {
        "indus": "khodro",
        "token": "خودرو",
        "name": "ایران خودرو",
        "fiscal_year": 12,
    },
    "kemina": {
        "indus": "shishe",
        "token": "کمینا",
        "name": "شیشه سازی مینا",
        "fiscal_year": 12,
    },
    "sheranol": {
        "indus": "ravankar",
        "token": "شرانل",
        "name": "نفت ایرانول",
        "fiscal_year": 12,
    },
    "ofogh": {
        "indus": "foroushgah",
        "token": "افق",
        "name": "فروشگاه های زنجیره ای افق",
        "fiscal_year": 12,
    },
    "kave": {
        "indus": "folad",
        "token": "کاوه",
        "name": "فولاد کاوه جنوب کیش",
        "fiscal_year": 12,
    },
    "hormoz": {
        "indus": "folad",
        "token": "هرمز",
        "name": "فولاد هرمزگان جنوب",
        "fiscal_year": 12,
    },
    "ghebshahr": {
        "indus": "ghaza",
        "token": "غبشهر",
        "name": "صنعتی بهشهر",
        "fiscal_year": 9,
    },
    "sharum": {
        "indus": "chemical",
        "token": "شاروم",
        "name": "پتروشیمی ارومیه",
        "fiscal_year": 12,
    },
    "jam": {
        "indus": "polymer",
        "token": "جم",
        "name": "پترو شیمی جم",
        "fiscal_year": 12,
    },
    "derazak": {
        "indus": "darou",
        "token": "درازک",
        "name": "لابراتوارهای رازک",
        "fiscal_year": 12,
    },
    "fasazan": {
        "indus": "folad",
        "token": "فسازان",
        "name": "غلتک سازان سپاهان",
        "fiscal_year": 12,
    },
    "sheguya": {
        "indus": "chemical",
        "token": "شگویا",
        "name": "پتروشیمی تندگویان",
        "fiscal_year": 12,
    },
    "sepaha": {
        "indus": "siman",
        "token": "سپاها",
        "name": "سیمان سپاهان",
        "fiscal_year": 6,
    },
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
indusries = {
    "folad": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "felezat": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "ravankar": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "dode": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "palayesh": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "urea": {
        "scenario": "dollar",
        "scenario_margin": "not",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "methanol": {
        "scenario": "dollar",
        "scenario_margin": "not",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "siman": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4 * 1.4,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "lastic": {
        "scenario": "last",
        "scenario_margin": "not",
        "alpha_rate": 1.3,
        "alpha_rate_next": 1.3 * 1.3,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "shishe": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4 * 1.4,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "kashi": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.4,
        "alpha_rate_next": 1.4 * 1.4,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "shir": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.2,
        "alpha_rate_next": 1.2 * 1.2,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "biscuit": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.35,
        "alpha_rate_next": 1.35 * 1.35,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "polymer": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "chemical": {
        "scenario": "dollar",
        "scenario_margin": "constant",
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "madani": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.35,
        "alpha_rate_next": 1.35 * 1.35,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "darou": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.30,
        "alpha_rate_next": 1.30 * 1.3,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "ghaza": {
        "scenario": "last",
        "scenario_margin": "constant",
        "alpha_rate": 1.30,
        "alpha_rate_next": 1.3 * 1.3,
        "energy_g": 1,
        "energy_g_next": 1,
    },
    "shoyande": {
        "scenario": "last",
        "scenario_margin": "not",
        "alpha_rate": 1.35,
        "alpha_rate_next": 1.35 * 1.35,
        "energy_g": 1,
        "energy_g_next": 1,
    },
}
pred_dollar = {1401: 380000, 1402: 450000, 1403: 650000}
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
