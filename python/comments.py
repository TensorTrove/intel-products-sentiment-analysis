import time
import logging
import csv
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract reviews from a page
def extract_reviews(driver):
    reviews = []
    review_elements = driver.find_elements(By.CSS_SELECTOR, ".review")
    for review_element in review_elements:
        review = {}
        try:
            # Extract the rating
            rating_element = review_element.find_element(By.CSS_SELECTOR, ".review-rating")
            review['rating'] = rating_element.text.strip()

            # Extract the review content
            review_content_element = review_element.find_element(By.CSS_SELECTOR, ".review-text-content")
            review['content'] = review_content_element.text.strip()

            # Extract the variant purchased
            try:
                variant_element = review_element.find_element(By.CSS_SELECTOR, ".review-format-strip")
                review['variant'] = variant_element.text.strip()
            except NoSuchElementException:
                review['variant'] = "No variant specified"

            # Extract the reviewer name
            name_element = review_element.find_element(By.CSS_SELECTOR, ".a-profile-name")
            review['name'] = name_element.text.strip()

            # Extract the review date
            date_element = review_element.find_element(By.CSS_SELECTOR, ".review-date")
            review['date'] = date_element.text.strip()

            # Determine if the review is verified or not
            try:
                verified_element = review_element.find_element(By.CSS_SELECTOR, ".a-color-state.a-text-bold")
                review['verified'] = 'Verified Purchase' in verified_element.text
            except NoSuchElementException:
                review['verified'] = False

            # Extract the sub-reviews (comments)
            comment_elements = review_element.find_elements(By.CSS_SELECTOR, ".review-comments .review-comment-content")
            sub_reviews = [comment_element.text.strip() for comment_element in comment_elements]
            review['sub_reviews'] = sub_reviews
        except NoSuchElementException as e:
            logging.error(f"Element not found: {e}")
            continue

        reviews.append(review)
    return reviews

# Function to navigate to the next page
def go_to_next_page(driver):
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, ".a-last a")
        next_button.click()
        return True
    except NoSuchElementException:
        return False

# Function to save reviews to a CSV file
def save_reviews_to_csv(reviews, filename):
    if reviews:
        keys = reviews[0].keys()
        if not os.path.exists(filename):
            with open(filename, 'w', newline='', encoding='utf-8') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(reviews)
        else:
            with open(filename, 'a', newline='', encoding='utf-8') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writerows(reviews)
        logging.info(f"Saved {len(reviews)} reviews to {filename}")
    else:
        logging.info("No reviews to save.")

# Main function
def main(url, num_pages, output_file, wait_time, retry_count, proxy):
    options = Options()
    options.add_argument("start-maximized")
    options.headless = True
    if proxy:
        options.add_argument(f'--proxy-server={proxy}')

    driver = webdriver.Chrome(options=options)

    all_reviews = []

    try:
        driver.get(url)
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".review"))
        )
        
        
        
        for page_number in range(num_pages):
            
            logging.info(f"Extracting reviews from page {page_number + 1}")
            retries = 0
            while retries < retry_count:
                try:
                    reviews = extract_reviews(driver)
                    logging.info(f"Found {len(reviews)} reviews on page {page_number + 1}")
                    all_reviews.extend(reviews)
                    break
                except TimeoutException:
                    retries += 1
                    logging.warning(f"Retry {retries}/{retry_count} for page {page_number + 1}")
                    time.sleep(wait_time)

            if retries == retry_count:
                logging.error(f"Failed to load page {page_number + 1} after {retry_count} retries")
                break

            if not go_to_next_page(driver):
                logging.info("No more pages to load.")
                break

            # Wait for the next page to load
            time.sleep(wait_time)
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".review"))
            )
    except TimeoutException:
        logging.error("Timeout waiting for reviews to load")
    finally:
        driver.quit()

    save_reviews_to_csv(all_reviews, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Amazon product reviews.")
    # parser.add_argument("url", help="The URL of the Amazon product review page")
    # parser.add_argument("--pages", type=int, default=1, help="Number of pages to scrape")
    parser.add_argument("--output", default="reviews.csv", help="Output CSV file")
    # parser.add_argument("--wait", type=int, default=10, help="Wait time in seconds for elements to load")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for loading a page")
    parser.add_argument("--proxy", help="Proxy server address")

    args = parser.parse_args()
    i=1
    with open("./links.txt","r") as f:
        for line in f:
            print(i)
            main(line, 15, args.output, 6, args.retries, args.proxy)
            i+=1