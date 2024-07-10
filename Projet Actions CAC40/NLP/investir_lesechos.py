from urllib.request import Request, urlopen
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import numpy as np

##################################################################################

headlines = []
newsflashes = []
keywords = []
categories = []
dates = []
links = []
errors = []

##################################################################################

def load_urls(company_name):
    return np.load(f"bdd/{company_name}_investir_lesechos.npy").tolist()

def source(url):
    request = Request(url , headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(request).read()
    return BeautifulSoup(webpage, "html.parser")

def headline(html):
    headlines.append(html.find("h1", class_="sc-14kwckt-6 sc-14omazk-0 sc-opo7ja-0 dJsezz fsUfny SfVAH sc-2c054-0 fyCDzT").text)

def newsflash(html):
    newsflashes.append(html.find("p", "sc-14kwckt-6 dbPXmO").text)
    
def date(html):
    published_at = html.find("span", class_="sc-17ifq26-0 fxAGFp").text.split()
    months = {"janv.": "01",
                "févr.": "02",
                "mars": "03",
                "avr.": "04",
                "mai": "05",
                "juin": "06",
                "juil.": "07",
                "août": "08",
                "sept.": "09",
                "oct.": "10",
                "nov.": "11",
                "déc.": "12"
            }
    day = published_at[2]
    month = published_at[3]
    year = published_at[4]
    if int(day)<10:
        day = f"0{day}"
    month = months[month]
    dates.append(datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y").date()) 

def link(url):
    links.append(url)

def has_same_length(list1, list2):
    if len(list1)>len(list2):
        list1.pop()
    elif len(list1)<len(list2):
        list2.pop()

def create_df():
    return pd.DataFrame({"headlines": headlines, "newsflashes": newsflashes, "dates": dates, "links": links})
    
def edit_df(df):
    df.sort_values(by = "dates", ascending = False, inplace = True)
    df.reset_index(drop = True, inplace = True)
    for i in range (len(df['dates'])):
        df['dates'][i] = df['dates'][i].strftime("%d/%m/%Y")
    return df

def save_df(df, company_name):
    df.to_csv(f"csv/{company_name}_investir_lesechos.csv")    

def main(company_name):
    company_name = company_name.replace(" ", "_")
    urls = load_urls(company_name)
    for url in urls:
        #html = source(url)
        request = Request(url , headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(request).read()
        html = BeautifulSoup(webpage, "html.parser")
        try:
            headline(html)
            newsflash(html)
            has_same_length(headlines, newsflashes)
            #keyword(html)
            #category(html)
            date(html)
            has_same_length(newsflashes, dates)
            link(url)
            has_same_length(dates, links)
        except AttributeError:
            errors.append(url)
    df = edit_df(create_df())
    save_df(df, company_name)