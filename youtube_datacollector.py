import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import config_by_name
from Basecollector import BaseCollector

# Configuration
env = os.getenv("FLASK_ENV", "default")
current_config = config_by_name[env]
api_key = current_config.YOUTUBE_API_KEY

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeDataCollector(BaseCollector):
    """Enhanced YouTube collector using BaseCollector architecture"""
    
    def __init__(self, rate_limit_delay: float = 2.0):
        super().__init__("YouTube", rate_limit_delay)
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("YouTube API key not found")
        
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("YouTube API client initialized")
        except Exception as e:
            raise ValueError(f"Failed to initialize YouTube API: {e}")
        
        # Educational content classification
        self.educational_keywords = {
            'primary': ['tutorial', 'course', 'learn', 'guide', 'how to', 'explained'],
            'secondary': ['fundamentals', 'basics', 'advanced', 'complete', 'step by step', 
                         'walkthrough', 'masterclass', 'bootcamp', 'training'],
            'academic': ['lecture', 'lesson', 'education', 'study', 'teach', 'instructor',
                        'professor', 'university', 'college', 'academy']
        }
        
        self.non_educational_indicators = [
            'reaction', 'reacts to', 'funny', 'fails', 'compilation', 'prank', 
            'challenge', 'vs', 'beef', 'drama', 'exposed', 'clickbait', 'shocking'
        ]
        
        self.duration_categories = {
            'micro_learning': (0, 5),   
            'short_tutorial': (5, 15),    
            'standard_lesson': (15, 45),   
            'long_form': (45, 120),       
            'course_length': (120, 999)    
        }
    
    def collect_courses(self, search_terms: List[str] = None, max_results_per_term: int = 10, 
                       include_all_durations: bool = True) -> List[Dict[str, Any]]:
        """Collect course data from YouTube (implements BaseCollector abstract method)"""
        
        if search_terms is None:
            search_terms = [
                "python programming beginner",
                "machine learning explained", 
                "web development full course",
                "sql database complete tutorial",
                "excel advanced tutorial"
            ]
        
        all_videos = []
        
        for term in search_terms:
            logger.info(f"Searching comprehensively for: {term}")
            
            if include_all_durations:
                for duration_type in ['short', 'medium', 'long']:
                    videos = self._search_by_duration(term, duration_type, max_results_per_term // 3)
                    all_videos.extend(videos)
            else:
                videos = self._search_by_duration(term, 'any', max_results_per_term)
                all_videos.extend(videos)
            
            self._rate_limit()
        
        unique_videos = self._remove_duplicates_and_enhance(all_videos)
        logger.info(f"Total unique videos collected: {len(unique_videos)}")
        return unique_videos
    
    def collect_engagement_data(self, course_id: str, **kwargs) -> Dict[str, Any]:
        """Collect engagement data for a specific YouTube video (implements BaseCollector abstract method)"""
        try:
            videos_response = self.youtube.videos().list(
                part='statistics,contentDetails',
                id=course_id
            ).execute()
            
            if not videos_response.get('items'):
                return {}
            
            video = videos_response['items'][0]
            statistics = video.get('statistics', {})
            
            view_count = int(statistics.get('viewCount', 0))
            like_count = int(statistics.get('likeCount', 0))
            comment_count = int(statistics.get('commentCount', 0))
            
            engagement_rate = (like_count + comment_count) / max(view_count, 1) * 100
            
            return {
                'views': view_count,
                'likes': like_count,
                'comments': comment_count,
                'engagement_rate': round(engagement_rate, 4)
            }
            
        except Exception as e:
            logger.error(f"Error collecting engagement data for {course_id}: {e}")
            return {}
    
    def _search_by_duration(self, search_term: str, duration: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for videos by duration category"""
        try:
            enhanced_query = f"{search_term} tutorial course guide complete"
            
            search_params = {
                'q': enhanced_query,
                'part': 'id,snippet',
                'type': 'video',
                'maxResults': min(max_results, 50),
                'order': 'relevance',
                'videoDefinition': 'any',
                'videoCaption': 'any',
                'safeSearch': 'moderate'
            }
            
            if duration != 'any':
                search_params['videoDuration'] = duration
            
            search_response = self.youtube.search().list(**search_params).execute()
            
            if not search_response.get('items'):
                return []
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            videos_response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails,status,topicDetails',
                id=','.join(video_ids)
            ).execute()
            
            videos = []
            for video in videos_response['items']:
                video_data = self._parse_enhanced_video_data(video, search_term, duration)
                if video_data:
                    videos.append(video_data)
            
            logger.info(f"Found {len(videos)} valid videos for '{search_term}' ({duration} duration)")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            if e.resp.status == 403:
                logger.error("API quota exceeded. Please try again later.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    
    def _parse_enhanced_video_data(self, video: Dict[str, Any], search_term: str, duration_filter: str) -> Optional[Dict[str, Any]]:
        """Parse video data into standardized format"""
        try:
            snippet = video['snippet']
            statistics = video.get('statistics', {})
            content_details = video.get('contentDetails', {})
            status = video.get('status', {})
            topic_details = video.get('topicDetails', {})
            
            if status.get('privacyStatus') != 'public':
                return None
            
            duration_info = self._parse_duration_enhanced(content_details.get('duration', ''))
            if not duration_info:
                return None
            
            duration_minutes, duration_seconds = duration_info
            
            title = snippet['title']
            description = snippet.get('description', '')
            tags = snippet.get('tags', [])
            
            educational_score = self._calculate_educational_score(title, description, tags)
            
            if educational_score < 0.3:
                return None
            
            if duration_minutes < 1:
                return None
            
            view_count = int(statistics.get('viewCount', 0))
            like_count = int(statistics.get('likeCount', 0))
            comment_count = int(statistics.get('commentCount', 0))
            
            engagement_rate = (like_count + comment_count) / max(view_count, 1) * 100
            likes_per_minute = like_count / max(duration_minutes, 1)
            comments_per_minute = comment_count / max(duration_minutes, 1)
            
            duration_category = self._categorize_duration(duration_minutes)
            channel_info = self._analyze_channel_educational_focus(snippet['channelTitle'], snippet['channelId'])
            published_at = self._parse_datetime(snippet['publishedAt'])
            complexity_score = self._estimate_content_complexity(title, description, duration_minutes)
            
            # Return data in BaseCollector expected format
            return {
                'title': title[:500],
                'instructor': snippet['channelTitle'][:200],
                'description': description[:2000],
                'url': f"https://www.youtube.com/watch?v={video['id']}",
                'thumbnail_url': snippet['thumbnails'].get('high', {}).get('url'),
                'duration_minutes': duration_minutes,
                'difficulty_level': self._estimate_difficulty_level(complexity_score),
                'category': search_term,
                'language': snippet.get('defaultLanguage', 'en'),
                'price': 0.0,
                'currency': 'USD',
                'is_free': True,
                'published_at': published_at,
                'tags': tags[:10] if tags else [],
                
                # Additional metadata for engagement metrics
                'metadata': {
                    'video_id': video['id'],
                    'search_term': search_term,
                    'duration_filter_used': duration_filter,
                    'collection_date': datetime.utcnow().isoformat(),
                    'api_version': 'v3_enhanced',
                    'educational_score': round(educational_score, 3),
                    'complexity_score': round(complexity_score, 3),
                    'duration_category': duration_category,
                    'channel_id': snippet['channelId'],
                    'channel_educational_focus': channel_info['educational_focus'],
                    'estimated_channel_authority': channel_info['authority_score'],
                    'view_count': view_count,
                    'like_count': like_count,
                    'comment_count': comment_count,
                    'engagement_rate': round(engagement_rate, 4),
                    'likes_per_minute': round(likes_per_minute, 2),
                    'comments_per_minute': round(comments_per_minute, 2),
                    'topic_categories': topic_details.get('topicCategories', [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing video data: {e}")
            return None
    
    def _parse_duration_enhanced(self, duration: str) -> Optional[tuple[float, int]]:
        """Parse ISO 8601 duration to minutes and seconds"""
        if not duration or not duration.startswith('PT'):
            return None
        
        try:
            duration = duration[2:]  
            hours = 0
            minutes = 0
            seconds = 0
            
            if 'H' in duration:
                hours = int(duration.split('H')[0])
                duration = duration.split('H')[1]
            if 'M' in duration:
                minutes = int(duration.split('M')[0])
                duration = duration.split('M')[1]
            if 'S' in duration:
                seconds = int(duration.split('S')[0])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            total_minutes = total_seconds / 60
            
            return (round(total_minutes, 2), total_seconds)
            
        except Exception:
            return None
    
    def _calculate_educational_score(self, title: str, description: str, tags: List[str]) -> float:
        """Calculate educational relevance score"""
        score = 0.0
        text_to_analyze = f"{title} {description} {' '.join(tags or [])}".lower()
        
        for keyword in self.educational_keywords['primary']:
            if keyword in text_to_analyze:
                score += 0.15
        
        for keyword in self.educational_keywords['secondary']:
            if keyword in text_to_analyze:
                score += 0.10
        
        for keyword in self.educational_keywords['academic']:
            if keyword in text_to_analyze:
                score += 0.12
        
        for indicator in self.non_educational_indicators:
            if indicator in text_to_analyze:
                score -= 0.20
        
        structured_indicators = ['chapter', 'part', 'lesson', 'module', 'section']
        for indicator in structured_indicators:
            if indicator in text_to_analyze:
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _categorize_duration(self, minutes: float) -> str:
        """Categorize video duration"""
        for category, (min_dur, max_dur) in self.duration_categories.items():
            if min_dur <= minutes < max_dur:
                return category
        return 'extended_course'
    
    def _analyze_channel_educational_focus(self, channel_title: str, channel_id: str) -> Dict[str, Any]:
        """Analyze channel's educational focus"""
        channel_lower = channel_title.lower()
        
        educational_focus = 0.5  
        authority_score = 0.5    
        
        edu_indicators = ['university', 'academy', 'school', 'education', 'learning', 
                         'tutorial', 'tech', 'programming', 'science', 'math']
        
        for indicator in edu_indicators:
            if indicator in channel_lower:
                educational_focus += 0.1
                authority_score += 0.05
        
        authority_indicators = ['official', 'certified', 'institute', 'foundation', 'lab']
        for indicator in authority_indicators:
            if indicator in channel_lower:
                authority_score += 0.15
        
        return {
            'educational_focus': min(1.0, educational_focus),
            'authority_score': min(1.0, authority_score)
        }
    
    def _estimate_content_complexity(self, title: str, description: str, duration: float) -> float:
        """Estimate content complexity for learning analytics"""
        complexity = 0.5  
        
        text = f"{title} {description}".lower()
        
        beginner_terms = ['beginner', 'intro', 'basic', 'fundamentals', 'getting started']
        intermediate_terms = ['intermediate', 'advanced beginner', 'next level']
        advanced_terms = ['advanced', 'expert', 'masterclass', 'deep dive', 'comprehensive']
        
        for term in beginner_terms:
            if term in text:
                complexity = max(0.2, complexity - 0.1)
        
        for term in intermediate_terms:
            if term in text:
                complexity = 0.6
        
        for term in advanced_terms:
            if term in text:
                complexity = min(0.9, complexity + 0.2)
        
        if duration > 60:
            complexity += 0.1
        elif duration > 120:
            complexity += 0.2
        
        return max(0.1, min(1.0, complexity))
    
    def _estimate_difficulty_level(self, complexity_score: float) -> str:
        """Convert complexity score to difficulty level"""
        if complexity_score < 0.4:
            return 'Beginner'
        elif complexity_score < 0.7:
            return 'Intermediate'
        else:
            return 'Advanced'
    
    def _remove_duplicates_and_enhance(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and enhance data"""
        seen_urls = set()
        unique_videos = []
        
        for video in videos:
            if video['url'] not in seen_urls:
                seen_urls.add(video['url'])
                unique_videos.append(video)
        
        logger.info(f"Removed {len(videos) - len(unique_videos)} duplicate videos")
        return unique_videos
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse ISO datetime string"""
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    
    def collect_and_save_with_metrics(self, search_terms: List[str] = None, max_results_per_term: int = 10) -> Dict[str, int]:
        """High-level method to collect courses and save with engagement metrics"""
        try:
            # Step 1: Collect course data
            courses_data = self.collect_courses(search_terms, max_results_per_term)
            
            if not courses_data:
                logger.warning("No courses collected")
                return {'saved': 0, 'metrics_saved': 0}
            
            # Step 2: Save courses using BaseCollector method
            course_ids = self.save_courses(courses_data)
            
            # Step 3: Save engagement metrics for each course
            metrics_saved = 0
            for course_data, course_id in zip(courses_data, course_ids):
                try:
                    # Extract engagement metrics from metadata
                    metadata = course_data.get('metadata', {})
                    
                    engagement_metrics = {
                        'views': metadata.get('view_count', 0),
                        'likes': metadata.get('like_count', 0),
                        'comments': metadata.get('comment_count', 0),
                        'engagement_rate': metadata.get('engagement_rate', 0),
                        'likes_per_minute': metadata.get('likes_per_minute', 0),
                        'comments_per_minute': metadata.get('comments_per_minute', 0),
                        'educational_score': metadata.get('educational_score', 0),
                        'complexity_score': metadata.get('complexity_score', 0)
                    }
                    
                    # Save metrics using BaseCollector method
                    self.save_engagement_metrics(course_id, engagement_metrics)
                    metrics_saved += 1
                    
                except Exception as e:
                    logger.error(f"Error saving metrics for course {course_id}: {e}")
                    continue
            
            logger.info(f"Successfully saved {len(course_ids)} courses and {metrics_saved} metric sets")
            
            return {
                'saved': len(course_ids),
                'metrics_saved': metrics_saved
            }
            
        except Exception as e:
            logger.error(f"Error in collect_and_save_with_metrics: {e}")
            return {'saved': 0, 'metrics_saved': 0}


def main():
    """Main execution function"""
    print("Enhanced YouTube Educational Content Collection Starting...")
    print("=" * 70)
    
    EDUCATIONAL_SEARCH_TERMS = [
        "python programming beginner",
        "machine learning explained",
        "web development full course",
        "sql database complete tutorial",
        "excel advanced tutorial",
        "digital marketing complete course",
        "project management fundamentals",
        "photoshop complete tutorial",
        "calculus step by step",
        "statistics explained simply"
    ]
    
    try:
        collector = YouTubeDataCollector(rate_limit_delay=2.0)
        
        # Use the new unified method
        results = collector.collect_and_save_with_metrics(
            EDUCATIONAL_SEARCH_TERMS, 
            max_results_per_term=8
        )
        
        print(f"✅ Collection completed successfully!")
        print(f"📚 Courses saved: {results['saved']}")
        print(f"📊 Metric sets saved: {results['metrics_saved']}")
        
        if results['saved'] > 0:
            print("\n🎯 Next steps:")
            print("1. Run analytics on your collected data")
            print("2. Build your Streamlit dashboard")
            print("3. Implement statistical analysis features")
        else:
            print("❌ No data was collected. Check your API key and network connection.")
            return 1
            
    except Exception as e:
        logger.error(f"Enhanced collection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)