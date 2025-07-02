import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import time
import logging
from Basecollector import BaseCollector
from datetime import datetime
import re
import sys
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class KhanAcademyCollector(BaseCollector):
    """Khan Academy sitemap-based scraper (filtered by topic)"""

    def __init__(self, rate_limit_delay: float = 1.5):
        super().__init__("Khan Academy", rate_limit_delay)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })

    def collect_courses(self) -> List[Dict[str, Any]]:
        sitemap_index_url = 'https://www.khanacademy.org/sitemap.xml'
        all_courses = []
        TARGET_DOMAINS = [
            'computing/computer-science', 
            'computing/programming', 
            'computing/hour-of-code', 
            'science/physics', 
            'science/chemistry', 
            'science/biology', 
            'college-admissions', 
            'math/calculus-1', 
            'math/linear-algebra', 
            'math/statistics-probability'
        ]

        try:
            logger.info(f"Fetching Khan Academy sitemap index: {sitemap_index_url}")
            index_response = self.session.get(sitemap_index_url)
            index_response.raise_for_status()
            index_root = ET.fromstring(index_response.text)

            sitemap_urls = [
                loc.text for loc in index_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if 'topic' in loc.text
            ]

            for sitemap_url in sitemap_urls:
                try:
                    logger.info(f"Fetching Khan Academy sitemap: {sitemap_url}")
                    sitemap_response = self.session.get(sitemap_url)
                    sitemap_response.raise_for_status()
                    root = ET.fromstring(sitemap_response.text)

                    for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                        loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                        if loc_elem is not None:
                            course_url = loc_elem.text
                            if any(f"/{domain}/" in course_url for domain in TARGET_DOMAINS):
                                all_courses.append({
                                    'title': course_url.split('/')[-1].replace('-', ' ').title(),
                                    'url': course_url,
                                    'instructor': 'Khan Academy',
                                    'category': 'general',
                                    'is_free': True,
                                    'metadata': {
                                        'course_type': 'topic',
                                        'scraped_at': datetime.now().isoformat(),
                                        'source': 'Khan Academy'
                                    }
                                })
                    self._rate_limit()
                except Exception as e:
                    logger.warning(f"Skipping sitemap due to error: {e}")
                    continue

            logger.info(f"Found {len(all_courses)} course entries from all sitemaps")

        except Exception as e:
            logger.error(f"Error scraping Khan Academy sitemap index: {e}")

        return all_courses

    def collect_engagement_data(self, course_url: str, **kwargs) -> Dict[str, Any]:
        return {}  # Khan Academy doesn't show public stats

def main():
    print("Starting Khan Academy Course Scraping")
    print("=" * 60)
    try:
        collector = KhanAcademyCollector()
        courses = collector.collect_courses()

        if not courses:
            print("No Khan Academy courses found.")
            return 1

        print(f"Collected {len(courses)} course listings")

        course_ids = collector.save_courses(courses)

        for course_data, course_id in zip(courses, course_ids):
            metrics = collector.collect_engagement_data(course_data['url'])
            collector.save_engagement_metrics(course_id, metrics)

        print("Khan Academy scraping complete.")
    except Exception as e:
        logger.error(f"Khan Academy data collection failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
