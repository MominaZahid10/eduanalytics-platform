import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
print("Loaded DB URL:", os.getenv("DATABASE_URL"))
class Config:
    # Common config
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")
    YOUTUBE_API_KEY=os.getenv("YOUTUBE_API_KEY")
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") 

    

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True  # ← Typo fixed: 'Testing' → 'TESTING'
    SQLALCHEMY_DATABASE_URI =os.getenv("DATABASE_URL")


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}
