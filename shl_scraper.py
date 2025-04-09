import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Dictionary mapping test type codes to their full forms
        self.test_type_map = {
            'A': 'Ability & Aptitude',
            'B': 'Biodata & Situational Judgement',
            'C': 'Competencies',
            'D': 'Development & 360',
            'E': 'Assessment Exercises',
            'K': 'Knowledge & Skills',
            'P': 'Personality & Behavior',
            'S': 'Simulations'
        }

    def _get_page_content(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return ""

    def _get_assessment_details(self, url: str) -> Dict:
        """Scrape detailed information from an assessment's page."""
        details = {
            'description': '',
            'job_levels': '',
            'languages': '',
            'assessment_length': ''
        }
        
        content = self._get_page_content(url)
        if not content:
            return details
            
        soup = BeautifulSoup(content, 'html.parser')
        
        # Get Description - updated to match the correct HTML structure
        desc_div = soup.find('div', class_='product-catalogue-training-calendar__row typ')
        if desc_div and desc_div.find('p'):
            details['description'] = desc_div.find('p').text.strip()
        
        # Get Job levels
        job_section = soup.find('h4', string='Job levels')
        if job_section and job_section.find_next('p'):
            details['job_levels'] = job_section.find_next('p').text.strip().rstrip(',')
        
        # Get Languages
        lang_section = soup.find('h4', string='Languages')
        if lang_section and lang_section.find_next('p'):
            details['languages'] = lang_section.find_next('p').text.strip().rstrip(',')
        
        # Get Assessment length
        length_section = soup.find('h4', string='Assessment length')
        if length_section and length_section.find_next('p'):
            length_text = length_section.find_next('p').text.strip()
            # Extract number from text like "Approximate Completion Time in minutes = 49"
            if match := re.search(r'=\s*(\d+)', length_text):
                details['assessment_length'] = match.group(1)
            else:
                details['assessment_length'] = length_text
        
        return details

    def _parse_catalog_page(self, html_content: str) -> List[Dict]:
        """Parse the HTML content and extract assessment information."""
        soup = BeautifulSoup(html_content, 'html.parser')
        assessments = []
        
        # Find all rows in the table
        rows = soup.find_all('tr')
        
        for row in rows:
            # Skip header rows
            if not row.find('a'):
                continue
                
            assessment = {}
            
            # Get assessment name and URL
            name_link = row.find('a')
            if name_link:
                assessment['name'] = name_link.text.strip()
                assessment['url'] = f"https://www.shl.com{name_link['href']}"
            
            # Get Remote Testing Support
            remote_testing = row.find_all('td')[1].find('span', class_='catalogue__circle -yes')
            assessment['remote_testing'] = 'Yes' if remote_testing else 'No'
            
            # Get Adaptive/IRT Support
            adaptive = row.find_all('td')[2].find('span', class_='catalogue__circle -yes')
            assessment['adaptive_irt'] = 'Yes' if adaptive else 'No'
            
            # Get Test Type
            test_type_cell = row.find_all('td')[3]
            test_type_spans = test_type_cell.find_all('span', class_='product-catalogue__key')
            test_types = []
            for span in test_type_spans:
                type_code = span.text.strip()
                if type_code in self.test_type_map:
                    test_types.append(self.test_type_map[type_code])
            assessment['test_type'] = ', '.join(test_types)
            
            assessments.append(assessment)
            
        return assessments

    def scrape_all(self):
        """Scrape all pages for both Individual Tests and Pre-packaged Solutions."""
        all_data = []
        
        # Scrape first two pages of Individual Test Solutions (type=1)
        for start in range(0, 373, 12):  # Changed from 373 to 24 to get first two pages
            logger.info(f"Scraping Individual Tests page {start//12 + 1}")
            content = self._get_page_content(f"{self.base_url}?start={start}&type=1&type=1")
            if content:
                page_data = self._parse_catalog_page(content)
                for item in page_data:
                    item['category'] = 'Individual Test Solutions'
                    # Get additional details from assessment page
                    logger.info(f"Fetching details for: {item['name']}")
                    details = self._get_assessment_details(item['url'])
                    item.update(details)
                all_data.extend(page_data)
            time.sleep(1)  # Be nice to the server
            
        # Scrape first two pages of Pre-packaged Job Solutions (type=2)
        for start in range(0, 133, 12):  # Changed from 133 to 24 to get first two pages
            logger.info(f"Scraping Pre-packaged Solutions page {start//12 + 1}")
            content = self._get_page_content(f"{self.base_url}?start={start}&type=2&type=2")
            if content:
                page_data = self._parse_catalog_page(content)
                for item in page_data:
                    item['category'] = 'Pre-packaged Job Solutions'
                    # Get additional details from assessment page
                    logger.info(f"Fetching details for: {item['name']}")
                    details = self._get_assessment_details(item['url'])
                    item.update(details)
                all_data.extend(page_data)
            time.sleep(1)  # Be nice to the server
            
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        # Reorder columns to put the new fields in a logical order
        columns = ['name', 'category', 'description', 'job_levels', 'languages', 
                  'assessment_length', 'remote_testing', 'adaptive_irt', 
                  'test_type', 'url']
        df = df[columns]
        df.to_csv('shl_assessments.csv', index=False)
        logger.info(f"Scraped {len(df)} total assessments")
        return df

if __name__ == "__main__":
    scraper = SHLScraper()
    results = scraper.scrape_all()
    print(f"Scraped {len(results)} assessments. Results saved to shl_assessments.csv")