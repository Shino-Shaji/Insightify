import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse,unquote
import os


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

base_url = f"https://www.amazon.in/Sony-WH-1000XM4-Cancelling-Headphones-Bluetooth/dp/B0863TXGM3?th=1"
parsed_url = urlparse(base_url)
path = parsed_url.path
path_parts = path.split("/")
product_name_encoded = path_parts[1]
product_name = unquote(product_name_encoded)

review_urls = get_review_urls(base_url)
reviews_df = extract_reviews(review_urls)
print(reviews_df)
save=input("save it to csv  y/n ?")
if save == "y":
    file_path = f"{product_name}.csv"
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, index_col=0)
        combined_df = pd.concat([existing_df, reviews_df], ignore_index=True)
        combined_df.to_csv(file_path)
    else:
        reviews_df.to_csv(file_path)
