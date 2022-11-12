import pickle
import smtplib, ssl
from email.message import EmailMessage

from statics.setting import DB, watchlist
from statics.secrets import sender, sender_pass, receiver
from Trade_Lib.boors_func import Stock

context = ssl.create_default_context()

em = EmailMessage()
em["From"] = sender
em["To"] = receiver
em["Subject"] = "Check Market Status"


with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender, sender_pass)

    data = {}
    alarm_text = []

    # create all stocks in watchlist
    for s in list(watchlist.keys())[:2]:
        stock = Stock(s)

        data[stock.Name] = stock
        stock_pe = stock.pe["P/E-ttm"].iloc[0]

        # alarm for good p/e
        if stock_pe < stock.pe["P/E-ttm"].median():
            alarm_text.append(f"PE {stock.Name} is {stock_pe}")

    # save data to pickle file
    with open(f"{DB}/data.pkl", "ab") as datafile:
        pickle.dump(data, datafile)

    # set text for sending email
    em.set_content(str(alarm_text))

    # send alarm with email
    server.sendmail(sender, receiver, em.as_string())
