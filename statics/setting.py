import os
from persiantools.jdatetime import JalaliDate


watchlist = {
    "fakhouz": {"indus": "folad", "token": "فخوز", "name": "فخوز"},
    "folad": {"indus": "folad", "token": "فولاد", "name": "فولاد مبارکه اصفهان"},
    "sekhouz": {"indus": "siman", "token": "سخوز", "name": "سیمان خوزستان"},
    "shekarbon": {"indus": "dode", "token": "شکربن", "name": "کربن ایران"},
    "kermasha": {"indus": "urea", "token": "کرماشا", "name": "صنایع پتروشیمی کرمانشاه"},
    "khorasan": {"indus": "urea", "token": "خراسان", "name": "پتروشیمی خراسان"},
    "shapdis": {"indus": "urea", "token": "شپدیس", "name": "پتروشیمی پردیس"},
    "shepoly": {"indus": "nasaji", "token": "شپلی", "name": "پلی اکریل ایران"},
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
    "ghefars": {"indus": "ghaza", "token": "غفارس", "name": " پگاه فارس"},
    "bemoto": {"indus": "bargh", "token": "بموتو", "name": "موتوژن"},
    "gheshasfa": {"indus": "ghaza", "token": "غشصفا", "name": "پگاه اصفهان"},
    "desobhan": {"indus": "darou", "token": "دسبحان", "name": "سبحان دارو"},
    "deshimi": {"indus": "darou", "token": "دشیمی", "name": "شیمی دارویی داروپخش"},
    "khedizel": {"indus": "khodro", "token": "خدیزل", "name": "بهمن دیزل"},
    "fameli": {"indus": "felezat", "token": "فملی", "name": " فملی "},
    "kechad": {"indus": "folad", "token": "کچاد", "name": "معدنی و صنعتی چادرملو"},
    "fasmin": {"indus": "felezat", "token": "فاسمین", "name": "کالسیمین"},
    "shebandar": {"indus": "palayesh", "token": "شبندر", "name": "پالایش نفت بندر"},
    "shetran": {"indus": "palayesh", "token": "شتران", "name": "پالایش نفت تهران"},
    "gheshan": {"indus": "ghaza", "token": "غشان", "name": "پگاه خراسان"},
    "deabid": {"indus": "darou", "token": "دعبید", "name": " دکتر عبیدی"},
    "dekapsul": {"indus": "darou", "token": "دکپسول", "name": "کپسول ایران"},
    "gharn": {"indus": "shoyande", "token": "قرن", "name": "پدیده شیمی قرن"},
<<<<<<< HEAD
    'shegol':{'indus':'shoyande','token':'شگل','name':'گلتاش'},
=======
    "shekolor": {"indus": "methanol", "token": "شکلر", "name": "نیروکلر"},
    "shegol": {"indus": "shoyande", "token": "شگل", "name": "گلتاش"},
>>>>>>> develop
}

ROOT_PATH = os.path.abspath(os.curdir).replace("\\", "/")
DB = f"{ROOT_PATH}/database"
WINDOWS_FIREFOX_DRIVER_PATH = f"{ROOT_PATH}/statics/geckodriver.exe"
LINUX_FIREFOX_DRIVER_PATH = f"{ROOT_PATH}/statics/geckodriver"

today = JalaliDate.today()
first_day = f"{today.year-10:02d}/{today.month:02d}/{today.day-1:02d}"
last_day = f"{today.year:02d}/{today.month:02d}/{today.day-1:02d}"

regex_per_timeid_y = "[۰۱۲۳۴۵۶۷۸۹]{4}/[۰۱۲۳۴۵۶۷۸۹]{2}/[۰۱۲۳۴۵۶۷۸۹]{2}"
regex_en_timeid_q = "[0123456789]{4}/[0123456789]{2}"
