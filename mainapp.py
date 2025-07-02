from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from config import config_by_name
from flask_migrate import Migrate


config_name=os.getenv('FLASK_ENV','development')
app=Flask(__name__)
app.config.from_object(config_by_name[config_name])
db=SQLAlchemy(app)
migrate = Migrate(app, db)

class Platform(db.Model):
    __tablename__="platforms"
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(80),unique=True,nullable=False)
    base_url=db.Column(db.String(200))
    is_active=db.Column(db.Boolean,default=True)
    created_at=db.Column(db.DateTime,default=datetime.utcnow)
    courses=db.relationship("Course",back_populates="platform")

    def __repr__(self):
        return f'<Platform{self.name}>'
    

class Course(db.Model):
    __tablename__ = 'courses'
    id = db.Column(db.Integer, primary_key=True)
    platform_id = db.Column(db.Integer, db.ForeignKey('platforms.id'), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    instructor = db.Column(db.String(200))
    description = db.Column(db.Text)
    url = db.Column(db.Text, unique=True)
    thumbnail_url = db.Column(db.Text)
    duration_minutes = db.Column(db.Integer)
    difficulty_level = db.Column(db.String(50))
    category = db.Column(db.String(100))
    language = db.Column(db.String(10))
    price = db.Column(db.Float)
    currency = db.Column(db.String(3))
    is_free = db.Column(db.Boolean, default=False)
    published_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    platform = db.relationship("Platform", back_populates="courses")
    engagement_metrics = db.relationship("EngagementMetric", back_populates="course")
    tags = db.Column(db.ARRAY(db.String))  


    def __repr__(self):
        return f'<Course {self.title[:30]}...>'

class EngagementMetric(db.Model):
    __tablename__ = 'engagement_metrics'
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    metric_type = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    course = db.relationship("Course", back_populates="engagement_metrics")


    def __repr__(self):
        return f'<Metric {self.metric_type}: {self.value}>'
    
def create_tables():
    """Create all database tables"""
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    print("Setting up EduAnalytics...")
    create_tables()
    print("\n Tables created. Ready to start your Flask app.")
    print(" Visit http://localhost:5000")
    app.run(debug=app.config['DEBUG'])
