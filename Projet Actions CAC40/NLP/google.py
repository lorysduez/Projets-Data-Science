from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
#from fake_useragent import UserAgent
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
#ua = UserAgent()
#userAgent = ua.random
#options.add_argument(f'user-agent={userAgent}')
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import numpy as np

##################################################################################

driver = webdriver.Chrome(executable_path = "chromedriver.exe", options = options)

links = []

##################################################################################

def research(company_name, website):
    url = f"https://www.google.com/search?q={company_name}+site:{website}"
    driver.get(url)
    try:    
        driver.find_element(By.ID, "L2AGLb").click() #Accept cookies
    except NoSuchElementException:
        pass

def get_links():
    articles = driver.find_elements(By.CLASS_NAME, "yuRUbf")
    for article in articles:
        links.append(article.find_element(By.TAG_NAME, "a").get_attribute("href"))

def has_next():
    try:
        driver.find_element(By.ID, "pnnext")
        return True
    except NoSuchElementException:
        return False

def next_page():
    next_page = driver.find_element(By.ID, "pnnext").get_attribute("href")
    driver.get(next_page)

def main(company_name, website):
    research(company_name, website)
    time.sleep(3)
    while has_next():
        get_links()
        next_page()
        time.sleep(5)
    company_name = company_name.replace(" ", "_")
    website = website.replace(".fr", "")
    np.save(f"bdd/{company_name}_{website}.npy", np.array(links))