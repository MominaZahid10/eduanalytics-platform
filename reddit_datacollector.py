import praw
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from Basecollector import BaseCollector
import re


logger = logging.getLogger(__name__)

class RedditCollector(BaseCollector):
    """Reddit API collector for educational discussions"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str, rate_limit_delay: float = 1.0):
        super().__init__("Reddit", rate_limit_delay)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def collect_courses(self, subreddits: List[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect course recommendations and discussions from Reddit"""
        if subreddits is None:
            subreddits = ['learnprogramming', 'MachineLearning', 'datascience', 
                         'webdev', 'cscareerquestions', 'OnlineLearning']
        
        all_courses = []
        
        for subreddit_name in subreddits:
            logger.info(f"Collecting from r/{subreddit_name}")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts related to courses
                for submission in subreddit.hot(limit=limit):
                    if self._is_course_related(submission.title, submission.selftext):
                        course_data = self._extract_course_mentions(submission)
                        if course_data:
                            all_courses.extend(course_data)
                
                self._rate_limit()
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                continue
        
        return all_courses
    
    def collect_engagement_data(self, submission_id: str, **kwargs) -> Dict[str, Any]:
        """Collect engagement data for a specific Reddit submission"""
        try:
            submission = self.reddit.submission(id=submission_id)
            
            return {
                'upvotes': submission.score,
                'comments': submission.num_comments,
                'upvote_ratio': submission.upvote_ratio,
                'gilded': submission.gilded,
                'discussion_engagement': self._calculate_discussion_score(submission)
            }
            
        except Exception as e:
            logger.error(f"Error collecting engagement data for {submission_id}: {e}")
            return {}
    
    def _is_course_related(self, title: str, text: str) -> bool:
        """Check if a post is related to online courses"""
        course_keywords = [
            'course', 'tutorial', 'learn', 'mooc', 'udemy', 'coursera', 
            'edx', 'pluralsight', 'youtube', 'free course', 'online course',
            'certification', 'bootcamp', 'nanodegree'
        ]
        
        content = f"{title} {text}".lower()
        return any(keyword in content for keyword in course_keywords)
    
    def _extract_course_mentions(self, submission) -> List[Dict[str, Any]]:
        """Extract course mentions from Reddit submission"""
        courses = []
        
        # Common course URL patterns
        url_patterns = [
            r'https?://(?:www\.)?coursera\.org/learn/[^\s]+',
            r'https?://(?:www\.)?udemy\.com/course/[^\s]+',
            r'https?://(?:www\.)?edx\.org/course/[^\s]+',
            r'https?://(?:www\.)?youtube\.com/(?:watch\?v=|playlist\?list=)[^\s]+',
            r'https?://(?:www\.)?pluralsight\.com/courses/[^\s]+',
            r'https?://(?:www\.)?khanacademy\.org/[^\s]+',
            r'https?://(?:www\.)?freecodecamp\.org/[^\s]+'
        ]
        
        content = f"{submission.title} {submission.selftext}"
        
        for pattern in url_patterns:
            matches = re.findall(pattern, content)
            for url in matches:
                course_data = {
                    'title': self._extract_title_from_context(submission.title, url),
                    'url': url,
                    'description': submission.selftext[:500] if submission.selftext else submission.title,
                    'category': self._infer_category(submission.subreddit.display_name),
                    'is_free': 'free' in content.lower(),
                    'metadata': {
                        'reddit_submission_id': submission.id,
                        'subreddit': submission.subreddit.display_name,
                        'reddit_score': submission.score,
                        'reddit_comments': submission.num_comments,
                        'posted_at': datetime.fromtimestamp(submission.created_utc).isoformat(),
                        'recommendation_context': submission.title
                    }
                }
                courses.append(course_data)
        
        return courses
    
    def _extract_title_from_context(self, submission_title: str, url: str) -> str:
        """Extract course title from Reddit context"""
        # Try to extract course name from URL
        url_parts = url.split('/')
        for part in url_parts:
            if len(part) > 10 and '-' in part:
                return part.replace('-', ' ').title()
        
        # Fallback to submission title
        return submission_title
    
    def _infer_category(self, subreddit_name: str) -> str:
        """Infer course category from subreddit"""
        category_map = {
            'learnprogramming': 'programming',
            'MachineLearning': 'machine_learning',
            'datascience': 'data_science',
            'webdev': 'web_development',
            'cscareerquestions': 'computer_science',
            'OnlineLearning': 'general'
        }
        return category_map.get(subreddit_name.lower(), 'general')
    
    def _calculate_discussion_score(self, submission) -> float:
        """Calculate a discussion engagement score"""
        try:
            # Load all comments
            submission.comments.replace_more(limit=0)
            
            total_comments = len(submission.comments.list())
            avg_comment_length = 0
            
            if total_comments > 0:
                total_length = sum(len(comment.body) for comment in submission.comments.list()[:50])
                avg_comment_length = total_length / min(total_comments, 50)
            
            # Engagement score based on comments, upvotes, and comment quality
            engagement_score = (
                (submission.score * 0.3) +
                (total_comments * 0.5) +
                (avg_comment_length / 10 * 0.2)
            )
            
            return min(engagement_score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Error calculating discussion score: {e}")
            return 0


def main():
    print(" Starting Reddit Educational Course Scraping")
    print("=" * 60)

    try:
        from config import config_by_name
        import os
        from dotenv import load_dotenv

        load_dotenv()

        env = os.getenv("FLASK_ENV", "default")
        current_config = config_by_name[env]

        client_id = current_config.CLIENT_ID
        client_secret = current_config.CLIENT_SECRET
        user_agent = current_config.USER_AGENT

        if not client_id or not client_secret or not user_agent:
            raise ValueError("Missing Reddit API credentials.")

        collector = RedditCollector(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            rate_limit_delay=1.0
        )

        courses = collector.collect_courses(limit=100)

        if not courses:
            print("No Reddit course posts found.")
            return 1

        print(f"Collected {len(courses)} Reddit course posts")

        course_ids = collector.save_courses(courses)

        for course_data, course_id in zip(courses, course_ids):
            submission_id = course_data["metadata"]["reddit_submission_id"]
            metrics = collector.collect_engagement_data(submission_id)
            collector.save_engagement_metrics(course_id, metrics)

        print("Reddit scraping complete.")

    except Exception as e:
        logger.error(f"Reddit data collection failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
