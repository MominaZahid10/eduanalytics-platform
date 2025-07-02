from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
import logging
from mainapp import app, db, Platform, Course, EngagementMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    
    def __init__(self, platform_name: str, rate_limit_delay: float = 1.0):
        self.platform_name = platform_name
        self.rate_limit_delay = rate_limit_delay
        self.platform_id = self._get_or_create_platform()
    
    def _get_or_create_platform(self) -> int:
        with app.app_context():
            platform = db.session.query(Platform).filter_by(name=self.platform_name).first()
            if not platform:
                platform = Platform(name=self.platform_name)
                db.session.add(platform)
                db.session.flush()
            return platform.id
    
    def _rate_limit(self):
        time.sleep(self.rate_limit_delay)
    
    @abstractmethod
    def collect_courses(self, **kwargs) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def collect_engagement_data(self, course_id: str, **kwargs) -> Dict[str, Any]:
        pass
    
    def save_courses(self, courses_data: List[Dict[str, Any]]) -> List[int]:
        course_ids = []
        with app.app_context():
            for course_data in courses_data:
                existing = db.session.query(Course).filter_by(url=course_data.get('url')).first()
                if existing:
                    logger.info(f"Course already exists: {course_data.get('title')}")
                    course_ids.append(existing.id)
                    continue
                
                course = Course(
                    platform_id=self.platform_id,
                    title=course_data.get('title'),
                    instructor=course_data.get('instructor'),
                    description=course_data.get('description'),
                    url=course_data.get('url'),
                    thumbnail_url=course_data.get('thumbnail_url'),
                    duration_minutes=course_data.get('duration_minutes'),
                    difficulty_level=course_data.get('difficulty_level'),
                    category=course_data.get('category'),
                    tags=course_data.get('tags'),
                    language=course_data.get('language'),
                    price=course_data.get('price'),
                    currency=course_data.get('currency'),
                    is_free=course_data.get('is_free', False),
                    published_at=course_data.get('published_at'),
                    metadata=course_data.get('metadata', {})
                )
                
                db.session.add(course)
                db.session.flush()
                course_ids.append(course.id)
                logger.info(f"Saved course: {course.title}")
            
            db.session.commit()
        
        return course_ids
    
    def save_engagement_metrics(self, course_id: int, metrics: Dict[str, Any]):
        with app.app_context():
            for metric_type, value in metrics.items():
                if value is not None:
                    metric = EngagementMetric(
                        course_id=course_id,
                        metric_type=metric_type,
                        value=float(value)
                    )
                    db.session.add(metric)
            db.session.commit()
            logger.info(f"Saved engagement metrics for course {course_id}")
