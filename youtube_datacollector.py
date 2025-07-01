import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import config_by_name
from collections import Counter

env = os.getenv("FLASK_ENV", "default")
current_config = config_by_name[env]

api_key = current_config.YOUTUBE_API_KEY
try:
    from mainapp import app, db, Platform, Course, EngagementMetric
except ImportError:
    print(" Error: Could not import Flask app and models.")
    sys.exit(1)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeDataCollector:
    def __init__(self):
        self.api_key=api_key
        if not self.api_key:
            raise ValueError("Youtube api key not found ")
        try:
            self.youtube=build('youtube','v3',developerKey=self.api_key)
            logger.info("Youtube api client initialized")
        except Exception as e:
            raise ValueError("Failed to initialize youtube api key")
        
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
        
        # Duration categories for comprehensive analysis

        self.duration_categories = {
            'micro_learning': (0, 5),   
            'short_tutorial': (5, 15),    
            'standard_lesson': (15, 45),   
            'long_form': (45, 120),       
            'course_length': (120, 999)    
        }
        with app.app_context():
            self.platform=Platform.query.filter_by(name="YouTube").first()
        if not self.platform:
            self.platform=Platform(
                name="YouTube",
                base_url="https://www.youtube.com",
                is_active=True
            )
            db.session.add(self.platform)
            db.session.commit()
            logger.info("Created youtube platform in database")

    def search_educational_content(self,search_terms:List[str],max_results_per_term:int=10,include_all_durations:bool=True)->List[Dict[str,Any]]:
        all_videos=[]
        for term in search_terms:
            logger.info("Searching comprehensively for:{term}")
            if include_all_durations:
                for duration_type in ['short','medium','long']:
                    videos=self.search_by_duration(term,duration_type,max_results_per_term//3)
            else:
                videos=self.search_by_duration(term,'any',max_results_per_term)
                all_videos.extend(videos)
            time.sleep(2)

        unique_videos=self._remove_duplicates_and_enhance(all_videos)
        logger.info(f"Total unique videos collected:{len(unique_videos)}")
        return unique_videos
        
    def search_by_duration(self, search_term: str, duration: str, max_results: int) -> List[Dict[str, Any]]:
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
            
            logger.info(f" Found {len(videos)} valid videos for '{search_term}' ({duration} duration)")
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
            
            return {
                'title': title[:500],
                'instructor': snippet['channelTitle'][:200],
                'description': description[:2000],
                'url': f"https://www.youtube.com/watch?v={video['id']}",
                'thumbnail_url': snippet['thumbnails'].get('high', {}).get('url'),
                
                'duration_minutes': duration_minutes,
                'duration_seconds': duration_seconds,
                'duration_category': duration_category,
                
                'category': search_term,
                'tags': tags[:15],  # More tags for better analysis
                'language': snippet.get('defaultLanguage', 'en'),
                'is_free': True,
                'published_at': published_at,
                
                'educational_score': round(educational_score, 3),
                'complexity_score': round(complexity_score, 3),
                
                'channel_id': snippet['channelId'],
                'channel_educational_focus': channel_info['educational_focus'],
                'estimated_channel_authority': channel_info['authority_score'],
                
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'engagement_rate': round(engagement_rate, 4),
                'likes_per_minute': round(likes_per_minute, 2),
                'comments_per_minute': round(comments_per_minute, 2),
                
                'topic_categories': topic_details.get('topicCategories', []),
                
                'metadata': {
                    'video_id': video['id'],
                    'search_term': search_term,
                    'duration_filter_used': duration_filter,
                    'collection_date': datetime.utcnow().isoformat(),
                    'api_version': 'v3_enhanced'
                }
            }
            
        except Exception as e:
            logger.error(f" Error parsing video data: {e}")
            return None
    
    def _parse_duration_enhanced(self, duration: str) -> Optional[tuple[float, int]]:
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
        for category, (min_dur, max_dur) in self.duration_categories.items():
            if min_dur <= minutes < max_dur:
                return category
        return 'extended_course'  
    
    def _analyze_channel_educational_focus(self, channel_title: str, channel_id: str) -> Dict[str, Any]:
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
    
    def _remove_duplicates_and_enhance(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        unique_videos = []
        
        for video in videos:
            if video['url'] not in seen_urls:
                seen_urls.add(video['url'])
                unique_videos.append(video)
        
        logger.info(f" Removed {len(videos) - len(unique_videos)} duplicate videos")
        return unique_videos
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    
    def save_to_database_enhanced(self, videos_data: List[Dict[str, Any]]) -> Dict[str, int]:
        with app.app_context():
            saved_count = 0
            skipped_count = 0
            updated_count = 0
            
            for video_data in videos_data:
                try:
                    existing = Course.query.filter_by(url=video_data['url']).first()
                    
                    if existing:
                        self._update_existing_course_metrics(existing, video_data)
                        updated_count += 1
                        continue
                    
                    course = Course(
                        platform_id=self.platform.id,
                        title=video_data['title'],
                        instructor=video_data['instructor'],
                        description=video_data['description'],
                        url=video_data['url'],
                        thumbnail_url=video_data['thumbnail_url'],
                        duration_minutes=video_data['duration_minutes'],
                        category=video_data['category'],
                        tags=','.join(video_data['tags']) if video_data['tags'] else None,
                        language=video_data['language'],
                        is_free=video_data['is_free'],
                        published_at=video_data['published_at']
                    )
                    
                    db.session.add(course)
                    db.session.flush() 
                    
                    metrics = [
                        EngagementMetric(course_id=course.id, metric_type='views', 
                                       value=video_data['view_count'], collected_at=datetime.utcnow()),
                        EngagementMetric(course_id=course.id, metric_type='likes', 
                                       value=video_data['like_count'], collected_at=datetime.utcnow()),
                        EngagementMetric(course_id=course.id, metric_type='comments', 
                                       value=video_data['comment_count'], collected_at=datetime.utcnow()),
                        EngagementMetric(course_id=course.id, metric_type='engagement_rate', 
                                       value=video_data['engagement_rate'], collected_at=datetime.utcnow()),
                        EngagementMetric(course_id=course.id, metric_type='likes_per_minute', 
                                       value=video_data['likes_per_minute'], collected_at=datetime.utcnow()),
                        EngagementMetric(course_id=course.id, metric_type='educational_score', 
                                       value=video_data['educational_score'], collected_at=datetime.utcnow()),
                        EngagementMetric(course_id=course.id, metric_type='complexity_score', 
                                       value=video_data['complexity_score'], collected_at=datetime.utcnow()),
                    ]
                    
                    for metric in metrics:
                        db.session.add(metric)
                    
                    saved_count += 1
                    
                    if saved_count % 10 == 0:
                        logger.info(f" Saved {saved_count} courses so far...")
                    
                except Exception as e:
                    logger.error(f" Error saving course '{video_data.get('title', 'Unknown')}': {e}")
                    db.session.rollback()
                    continue
            
            try:
                db.session.commit()
                logger.info(f"Database operation completed!")
                return {
                    'saved': saved_count,
                    'skipped': skipped_count,
                    'updated': updated_count
                }
            except Exception as e:
                logger.error(f" Database commit failed: {e}")
                db.session.rollback()
                return {'saved': 0, 'skipped': 0, 'updated': 0}
    
    def _update_existing_course_metrics(self, course, video_data):
        try:
            # Add new metric entries with current timestamp
            new_metrics = [
                EngagementMetric(course_id=course.id, metric_type='views', 
                               value=video_data['view_count'], collected_at=datetime.utcnow()),
                EngagementMetric(course_id=course.id, metric_type='likes', 
                               value=video_data['like_count'], collected_at=datetime.utcnow()),
                EngagementMetric(course_id=course.id, metric_type='engagement_rate', 
                               value=video_data['engagement_rate'], collected_at=datetime.utcnow()),
            ]
            
            for metric in new_metrics:
                db.session.add(metric)
                
        except Exception as e:
            logger.error(f" Error updating metrics for course {course.id}: {e}")
    

def main():
    print(" Enhanced YouTube Educational Content Collection Starting...")
    print("="*70)
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
        collector = YouTubeDataCollector()
        
        videos = collector.search_educational_content(
            EDUCATIONAL_SEARCH_TERMS, 
            max_results_per_term=10,  
            include_all_durations=True
        )
        
        if not videos:
            print(" No educational videos found")
            return 1
        
        print(f" Collected {len(videos)} unique educational videos")
        
        print(" Saving to database with enhanced analytics...")
        results = collector.save_to_database_enhanced(videos)
            
    except Exception as e:
        logger.error(f" Enhanced collection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
