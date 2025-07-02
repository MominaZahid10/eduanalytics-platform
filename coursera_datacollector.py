import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import time
import logging
from Basecollector import BaseCollector
import re
from datetime import datetime
import sys
logger = logging.getLogger(__name__)

class CourseraCollector(BaseCollector):
    """Coursera web scraper"""
    
    def __init__(self, rate_limit_delay: float = 2.0):
        super().__init__("Coursera", rate_limit_delay)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def collect_courses(self, categories: List[str] = None, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Collect courses from Coursera browse pages"""
        if categories is None:
            categories = ['data-science', 'computer-science', 'business', 'personal-development']
        
        all_courses = []
        
        for category in categories:
            logger.info(f"Scraping Coursera category: {category}")
            
            for page in range(1, max_pages + 1):
                try:
                    url = f"https://www.coursera.org/search?query={category}&page={page}"
                    response = self.session.get(url)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    courses = self._parse_course_listings(soup, category)
                    
                    if not courses:
                        logger.info(f"No more courses found for {category} on page {page}")
                        break
                    
                    all_courses.extend(courses)
                    logger.info(f"Found {len(courses)} courses on page {page} of {category}")
                    
                    self._rate_limit()
                    
                except Exception as e:
                    logger.error(f"Error scraping Coursera page {page} for {category}: {e}")
                    continue
        
        return all_courses
    
    def collect_engagement_data(self, course_url: str, **kwargs) -> Dict[str, Any]:
        """Collect engagement data from individual course pages"""
        try:
            response = self.session.get(course_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract ratings and enrollment data
            rating = self._extract_rating(soup)
            enrollments = self._extract_enrollments(soup)
            reviews_count = self._extract_reviews_count(soup)
            
            return {
                'rating': rating,
                'enrollments': enrollments,
                'reviews': reviews_count
            }
            
        except Exception as e:
            logger.error(f"Error collecting engagement data from {course_url}: {e}")
            return {}
    
    def _parse_course_listings(self, soup: BeautifulSoup, category: str) -> List[Dict[str, Any]]:
        """Parse course listings from Coursera browse page"""
        courses = []
        
        # Coursera uses dynamic class names, so we need to be flexible
        course_cards = soup.find_all(['div', 'article'], class_=re.compile(r'cds-.*card|ProductCard|SearchCard'))
        
        if not course_cards:
            # Try alternative selectors
            course_cards = soup.find_all('div', attrs={'data-track-component': 'browse_card'})
        
        for card in course_cards:
            try:
                course_data = self._extract_course_data(card, category)
                if course_data:
                    courses.append(course_data)
            except Exception as e:
                logger.debug(f"Error parsing course card: {e}")
                continue
        
        return courses
    
    def _extract_course_data(self, card, category: str) -> Optional[Dict[str, Any]]:
        """Extract course data from a single course card"""
        try:
            # Title and URL
            title_link = card.find('a', href=re.compile(r'/learn/'))
            if not title_link:
                return None
            
            title = title_link.get_text(strip=True)
            relative_url = title_link.get('href')
            full_url = f"https://www.coursera.org{relative_url}" if relative_url.startswith('/') else relative_url
            
            # Instructor/Institution
            instructor_elem = card.find(['span', 'div', 'p'], string=re.compile(r'University|College|Institute|School')) or \
                            card.find(['span', 'div', 'p'], class_=re.compile(r'partner|instructor|institution'))
            instructor = instructor_elem.get_text(strip=True) if instructor_elem else "Unknown"
            
            # Rating
            rating_elem = card.find(['span', 'div'], class_=re.compile(r'rating|star'))
            rating = None
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    rating = float(rating_match.group(1))
            
            # Difficulty level
            difficulty_elem = card.find(['span', 'div'], string=re.compile(r'Beginner|Intermediate|Advanced|Mixed'))
            difficulty = difficulty_elem.get_text(strip=True) if difficulty_elem else None
            
            # Duration
            duration_elem = card.find(['span', 'div'], string=re.compile(r'\d+\s*(week|month|hour)'))
            duration_text = duration_elem.get_text(strip=True) if duration_elem else None
            duration_minutes = self._parse_duration_text(duration_text)
            
            # Check if it's free
            free_elem = card.find(['span', 'div'], string=re.compile(r'Free|Audit|No cost'))
            is_free = bool(free_elem)
            
            return {
                'title': title,
                'instructor': instructor,
                'url': full_url,
                'category': category,
                'difficulty_level': difficulty,
                'duration_minutes': duration_minutes,
                'is_free': is_free,
                'metadata': {
                    'platform_rating': rating,
                    'course_type': 'course',  # vs specialization, certificate
                    'scraped_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.debug(f"Error extracting course data: {e}")
            return None
    
    def _parse_duration_text(self, duration_text: str) -> Optional[int]:
        """Parse duration text to minutes"""
        if not duration_text:
            return None
        
        # Extract numbers and time units
        matches = re.findall(r'(\d+)\s*(week|month|hour|day)', duration_text.lower())
        
        total_minutes = 0
        for number, unit in matches:
            number = int(number)
            if unit == 'hour':
                total_minutes += number * 60
            elif unit == 'day':
                total_minutes += number * 24 * 60
            elif unit == 'week':
                total_minutes += number * 7 * 24 * 60
            elif unit == 'month':
                total_minutes += number * 30 * 24 * 60
        
        return total_minutes if total_minutes > 0 else None
    
    def _extract_rating(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract course rating from course page"""
        rating_elem = soup.find(['span', 'div'], class_=re.compile(r'rating|star'))
        if rating_elem:
            rating_text = rating_elem.get_text(strip=True)
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                return float(rating_match.group(1))
        return None
    
    def _extract_enrollments(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract enrollment count from course page"""
        enrollment_elem = soup.find(['span', 'div'], string=re.compile(r'\d+[,\d]*\s*(student|learner|enroll)'))
        if enrollment_elem:
            enrollment_text = enrollment_elem.get_text(strip=True)
            enrollment_match = re.search(r'([\d,]+)', enrollment_text)
            if enrollment_match:
                return int(enrollment_match.group(1).replace(',', ''))
        return None
    
    def _extract_reviews_count(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract review count from course page"""
        review_elem = soup.find(['span', 'div'], string=re.compile(r'\d+[,\d]*\s*review'))
        if review_elem:
            review_text = review_elem.get_text(strip=True)
            review_match = re.search(r'([\d,]+)', review_text)
            if review_match:
                return int(review_match.group(1).replace(',', ''))
        return None
def main():
    print("Starting Coursera Educational Course Scraping")
    print("=" * 60)
    
    try:
        collector = CourseraCollector(rate_limit_delay=2.0)
        
        # Step 1: Scrape listings from categories
        courses = collector.collect_courses(
            categories = ['python', 'machine learning', 'artificial intelligence', 'deep learning', 'data science'] , 
            max_pages=3
        )

        if not courses:
            print("No Coursera courses found.")
            return 1
        
        print(f"Collected {len(courses)} course listings")
        
        # Step 2: Save course metadata to database
        course_ids = collector.save_courses(courses)
        
        # Step 3: Visit course pages and collect engagement metrics
        for course_data, course_id in zip(courses, course_ids):
            metrics = collector.collect_engagement_data(course_data['url'])
            collector.save_engagement_metrics(course_id, metrics)

        print(" Coursera scraping complete.")
    
    except Exception as e:
        logger.error(f" Coursera data collection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)