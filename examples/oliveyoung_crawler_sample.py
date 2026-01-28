import requests
from bs4 import BeautifulSoup
import time
import csv
import os
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from itertools import zip_longest
from selenium import webdriver
from selenium.webdriver.common.by import By
  
def load_data():
    data = []
    with open("./data/suncream_list.csv") as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            data.append(row)
    return data
  
def write_data(data):
    file_path = "./data/suncream_reviews_score.csv"
    file_exists = os.path.isfile("./data/suncream_reviews_score.csv")
    
    with open(file_path, "a", newline='', encoding='utf-8') as fw:
        writer = csv.DictWriter(fw, fieldnames=["page", "product_name", "star", "title", "review", "skin_type"])
        
        # 파일이 존재하지 않으면 헤더를 작성합니다.
        if not file_exists:
            writer.writeheader()
        
        for row in data:
            writer.writerow(row)

def crawl_parse_review_html_write_data(url):
    
    driver = webdriver.Chrome()
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reviewInfo"]'))).click()
        time.sleep(3)

        for page_num in range(1, 200):
            parse_review_text_list = []
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')

            product_name_tag = soup.find("p", class_="prd_name")
            product_name = product_name_tag.text if product_name_tag else []

            user_clrfix_tags = soup.find_all("div", class_="user clrfix")
            review_tags = soup.find_all("div", class_="txt_inner")
            title_tags = soup.find_all("div", class_="poll_sample")
            star_tags = soup.find_all("div", class_ = "score_area")

            combined_list = list(zip_longest(title_tags, review_tags, user_clrfix_tags, star_tags, fillvalue=None))

            for title_tag, review_tag, user_clrfix_tag, star_tag in combined_list:
                review_text = review_tag.text if review_tag else None
                title_text = [tag.text.strip() for tag in title_tag.find_all("span")[1::2]] if title_tag else None
                span_text = [span.text.strip() for span in user_clrfix_tag.find_all("span")[1:]] if user_clrfix_tag else None
                star_text = [star.text.strip() for star in star_tag.find_all("span")][0]

                review_data = {
                    "page": page_num,
                    "product_name": product_name,
                    "star" : star_text,
                    "title": title_text,
                    "review": review_text,
                    "skin_type": span_text
                }
                parse_review_text_list.append(review_data)

            write_data(parse_review_text_list)

            try:
                next_button = driver.find_element(By.XPATH, f"//a[@data-page-no='{page_num + 1}']")
                next_button.click()
                time.sleep(3)
            except NoSuchElementException:
                print(f"Page {page_num} is the last page. \n ")
                break

    finally:
        driver.quit()
    
    return parse_review_text_list

if __name__ == '__main__':
    data = load_data()
    parse_review_list = []
    
    for i, review in enumerate(data):
        url = review["product_link"]
        try:
            parse_review = crawl_parse_review_html_write_data(url)
        except:
            continue
        print(f"{i}번 제품 끝")