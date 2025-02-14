import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse,unquote
import google.generativeai as genai
import json

with open('C:/Users/shino/Desktop/Amzn Scraper UI/revscr/scraper/api_keys.json') as f:
    api_keys = json.load(f)


def extract_reviews(urls):
    all_reviews = []

    for url in urls:
        print(f"Fetching reviews from: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        reviews = soup.find_all('div', {'data-hook': 'review'})
        print(f"Found {len(reviews)} reviews on this page")

        for review in reviews:
            try:
                review_data = {}
                review_title_element = review.find('a', class_='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold')
                review_data['review_title'] = review_title_element.text.split('\n')[1].strip() if review_title_element else ''
                review_data['reviewers_name'] = review.find('span', class_='a-profile-name').text.strip()
                review_data['review_body'] = review.find('span', class_='a-size-base review-text review-text-content').text.strip()
                review_data['star_rating'] = float(review.find('span', class_='a-icon-alt').text.split()[0])
                review_data['review_date'] = review.find('span', class_='a-size-base a-color-secondary review-date').text.strip()

                review_date_text = review_data['review_date']
                pattern = r'\d+\s\w+\s\d+'
                date_match = re.search(pattern, review_date_text)
                if date_match:
                    review_date = date_match.group()
                    review_data['review_date']=review_date
                else:
                    print("Date not found in the review data.")
                
                helpful_votes_element = review.find('span', class_='a-size-base a-color-tertiary cr-vote-text')
                helpful_votes_text = helpful_votes_element.text if helpful_votes_element else None
                helpful_votes = None
                if helpful_votes_text:
                    helpful_votes_match = re.search(r'(\d+)', helpful_votes_text)
                    if helpful_votes_match:
                        helpful_votes = int(helpful_votes_match.group())
                
                review_data['helpful_votes'] = helpful_votes if helpful_votes is not None else 0

                all_reviews.append(review_data)
            except Exception as e:
                print(f"Error occurred while processing review: {e}")

    df = pd.DataFrame(all_reviews)
    return df

def get_review_urls(base_url):
    urls = []
    page_number = 1

    while page_number==1:
        updated_url = f"{base_url}&pageNumber={page_number}"
        print(f"Fetching reviews from page {page_number}: {updated_url}")
        response = requests.get(updated_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        reviews_section = soup.find('div', {'id': 'cm_cr-review_list'})
        if not reviews_section:
            print("No reviews found on this page. Exiting pagination.")
            break

        urls.append(updated_url)
        page_number += 1

    return urls

GOOGLE_API_KEY = api_keys.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model=genai.GenerativeModel('gemini-pro')

base_url = "https://www.amazon.in/Apple-AirPods-Generation-MagSafe-USB%E2%80%91C/product-reviews/B0CHX719JD/?ie=UTF8&reviewerType=all_reviews&pageNumber=1"
parsed_url = urlparse(base_url)
path = parsed_url.path
path_parts = path.split("/")
product_name_encoded = path_parts[1]
product_name = unquote(product_name_encoded)

import pandas as pd

csv_files = ["1.csv", "2.csv", "3.csv","4.csv","5.csv","6.csv"]  

all_data = []

for filename in csv_files:
    df = pd.read_csv(filename)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(f'{product_name}', index=False)