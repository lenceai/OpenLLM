"""
LinkedIn crawler for collecting user's posts and articles.
"""

import time
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .base_crawler import BaseCrawler

class LinkedInCrawler(BaseCrawler):
    """Crawler for LinkedIn profiles and posts."""
    
    def __init__(self, url: str):
        """
        Initialize the LinkedIn crawler.
        
        Args:
            url: The LinkedIn profile URL.
        """
        super().__init__(url)
        self.profile_url = url
        self.driver = None
    
    def _setup_driver(self):
        """Set up the Selenium WebDriver with Chrome options."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
    
    def authenticate(self, credentials: Optional[Dict[str, str]] = None) -> bool:
        """
        Authenticate with LinkedIn.
        
        Args:
            credentials: Dictionary containing 'username' and 'password'.
            
        Returns:
            True if authentication was successful, False otherwise.
        """
        if not credentials or 'username' not in credentials or 'password' not in credentials:
            self.logger.error("LinkedIn credentials not provided")
            return False
        
        if not self.driver:
            self._setup_driver()
        
        try:
            # Navigate to LinkedIn login page
            self.driver.get("https://www.linkedin.com/login")
            
            # Enter username and password
            username_field = self.driver.find_element(By.ID, "username")
            password_field = self.driver.find_element(By.ID, "password")
            
            username_field.send_keys(credentials['username'])
            password_field.send_keys(credentials['password'])
            
            # Click the login button
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for redirection to complete
            WebDriverWait(self.driver, 10).until(
                EC.url_contains("linkedin.com/feed")
            )
            
            self.logger.info("Successfully authenticated with LinkedIn")
            return True
            
        except TimeoutException:
            self.logger.error("Authentication timed out")
            return False
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def crawl(self) -> List[Dict[str, Any]]:
        """
        Crawl the LinkedIn profile and collect posts.
        
        Returns:
            A list of dictionaries containing post data.
        """
        if not self.driver:
            self._setup_driver()
        
        raw_data = []
        
        try:
            # Navigate to profile page
            self.driver.get(self.profile_url)
            
            # Wait for profile to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "pv-top-card"))
            )
            
            # Scroll down to load more posts
            self._scroll_page()
            
            # Extract posts
            posts = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'feed-shared-update-v2')]")
            
            for post in posts:
                try:
                    # Extract post content
                    post_text_element = post.find_element(By.XPATH, ".//div[contains(@class, 'feed-shared-update-v2__description')]//span")
                    post_text = post_text_element.text
                    
                    # Extract timestamp
                    timestamp_element = post.find_element(By.XPATH, ".//span[contains(@class, 'visually-hidden') and contains(text(), 'ago')]")
                    timestamp = timestamp_element.text
                    
                    # Extract likes count if available
                    try:
                        likes_element = post.find_element(By.XPATH, ".//span[contains(@class, 'social-details-social-counts__reactions-count')]")
                        likes_count = likes_element.text
                    except:
                        likes_count = "0"
                    
                    post_data = {
                        "platform": "linkedin",
                        "type": "post",
                        "content": post_text,
                        "timestamp": timestamp,
                        "metadata": {
                            "likes": likes_count
                        }
                    }
                    
                    raw_data.append(post_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting post data: {str(e)}")
                    continue
            
            # Now let's try to get articles if any
            try:
                # Navigate to articles tab
                articles_tab = self.driver.find_element(By.XPATH, "//a[contains(@href, '/detail/recent-activity/posts/')]")
                articles_tab.click()
                
                # Wait for articles to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'ember-view artdeco-card')]"))
                )
                
                # Extract articles
                articles = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'ember-view artdeco-card')]")
                
                for article in articles:
                    try:
                        # Extract article title
                        title_element = article.find_element(By.XPATH, ".//h2[contains(@class, 'feed-shared-article__title')]")
                        title = title_element.text
                        
                        # Extract article link
                        link_element = article.find_element(By.XPATH, ".//a[contains(@class, 'feed-shared-article__meta-link')]")
                        link = link_element.get_attribute("href")
                        
                        # Visit the article to get full content
                        self.driver.get(link)
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'article-content')]"))
                        )
                        
                        # Extract full article content
                        content_element = self.driver.find_element(By.XPATH, "//div[contains(@class, 'article-content')]")
                        content = content_element.text
                        
                        article_data = {
                            "platform": "linkedin",
                            "type": "article",
                            "title": title,
                            "content": content,
                            "url": link,
                            "metadata": {}
                        }
                        
                        raw_data.append(article_data)
                        
                    except Exception as e:
                        self.logger.warning(f"Error extracting article data: {str(e)}")
                        continue
                    
            except Exception as e:
                self.logger.warning(f"Error accessing articles tab: {str(e)}")
            
            self.logger.info(f"Collected {len(raw_data)} items from LinkedIn")
            
        except Exception as e:
            self.logger.error(f"Error during LinkedIn crawling: {str(e)}")
        
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
        
        return raw_data
    
    def _scroll_page(self, scroll_pause_time: float = 2.0, num_scrolls: int = 5):
        """
        Scroll down the page to load more content.
        
        Args:
            scroll_pause_time: Time to pause between scrolls.
            num_scrolls: Number of times to scroll.
        """
        for _ in range(num_scrolls):
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait to load page
            time.sleep(scroll_pause_time)
    
    def standardize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Standardize LinkedIn data to a common format.
        
        Args:
            raw_data: Raw data collected from LinkedIn.
            
        Returns:
            A list of dictionaries with standardized data.
        """
        standardized_data = []
        
        for item in raw_data:
            standardized_item = {
                "source": "linkedin",
                "type": item.get("type", "post"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", self.profile_url),
                "timestamp": item.get("timestamp", ""),
                "metadata": item.get("metadata", {})
            }
            
            standardized_data.append(standardized_item)
        
        return standardized_data 