from urllib.request import Request, urlopen
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
    return np.load(f"bdd/{company_name}_latribune.npy").tolist()

def select_urls(all_urls, company_name):
    latribune_urls = []
    bourse_latribune_urls = []
    other_latribune_urls = []
    for url in all_urls:
        if "www.latribune.fr" in url:
            latribune_urls.append(url)
        elif "bourse.latribune.fr" in url:
            bourse_latribune_urls.append(url)
        else:
            other_latribune_urls.append(url)
    np.save(f"bdd/{company_name}_bourse_latribune.npy", np.array(bourse_latribune_urls))
    np.save(f"bdd/{company_name}_other_latribune.npy", np.array(other_latribune_urls))
    return latribune_urls

def source(url):
    request = Request(url , headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(request).read()
    return BeautifulSoup(webpage, "html.parser")

def headline(html):
    headlines.append(html.find("h1", itemprop = "Headline").text)
    
def newsflash(html):
    newsflashes.append(html.section.contents[0].lstrip().rstrip())
    
"""
def keyword(url):
    words = source(url).find_all("a", class_="sc-ap0kf6-0 JCNOU")
    words_list = []
    for word in words:
        words_list.append(word.text)
    keywords.append(words_list)

def category(url):
    categories.append(source(url).find("a", class_="sc-mzdmh8-0 kXwCnh sc-ztp7xd-0 cqpgii active").text)
"""
def date(html):
    article_date = html.find("time", itemprop = "datePublished").attrs["datetime"][:10]
    dates.append(datetime.strptime(article_date, "%Y-%m-%d").date())

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
    df.to_csv(f"csv/{company_name}_latribune.csv")

def main(company_name):
    company_name = company_name.replace(" ", "_")
    urls = select_urls(load_urls(company_name), company_name)
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