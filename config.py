import os
from dotenv import load_dotenv

load_dotenv()
print("Loaded DB URL:", os.getenv("DATABASE_URL"))
class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")
    YOUTUBE_API_KEY=os.getenv("YOUTUBE_API_KEY")
    CLIENT_ID=os.getenv("client_id")
    CLIENT_SECRET=os.getenv("client_secret")
    USER_AGENT=os.getenv("user_agent")
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") 

    

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True  
    SQLALCHEMY_DATABASE_URI =os.getenv("DATABASE_URL")


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}
