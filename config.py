import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

def get_database_url():
    """Get database URL from Streamlit secrets (cloud) or environment variables (local)"""
    try:
        import streamlit as st
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets["DATABASE_URL"]
    except:
        # Fallback to environment variables (for local development)
        return os.getenv("DATABASE_URL")

def get_secret(key, default=None):
    """Get secret from Streamlit secrets (cloud) or environment variables (local)"""
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except:
        return os.getenv(key, default)

# Get database URL
DATABASE_URL = get_database_url()
print("Loaded DB URL:", DATABASE_URL[:50] + "..." if DATABASE_URL else "None")

class Config:
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = get_secret("SECRET_KEY", "fallback-secret-key")
    YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")
    CLIENT_ID = get_secret("CLIENT_ID")
    CLIENT_SECRET = get_secret("CLIENT_SECRET")
    USER_AGENT = get_secret("USER_AGENT")
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = DATABASE_URL

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = DATABASE_URL

config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}