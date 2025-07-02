import sys
from datetime import datetime

try:
    from mainapp import app,db,Platform
except ImportError:
    print("Error: Could not import Flask app and models.")
    sys.exit(1)

def setup_database():
    with app.app_context():
        try:
          db.create_all(
          )
          platforms_data=[
              {
                  "name": "YouTube",
                  "base_url":"https://www.youtube.com",
                  "is_active":False,
                  "description":"World's largest video sharing platform with educational content"
              },
              {
                    "name": "Coursera",
                    "base_url": "https://www.coursera.org",
                    "is_active": False,
                    "description": "Online courses from top universities and companies"
                },
                {
                    "name": "Khan Academy",
                    "base_url": "https://www.khanacademy.org",
                    "is_active": False,
                    "description": "Free online courses, lessons and practice"
                },
                {
                     "name": "Reddit",
                     "base_url": "https://www.reddit.com",
                     "is_active": False,
                     "description": "Community-driven discussions and course recommendations"
}

          ]
          added_count = 0
          for platform_info in platforms_data:
                existing = Platform.query.filter_by(name=platform_info["name"]).first()
                
                if not existing:
                    platform = Platform(
                        name=platform_info["name"],
                        base_url=platform_info["base_url"],
                        is_active=platform_info["is_active"]
                    )
                    
                    if hasattr(Platform, 'description'):
                        platform.description = platform_info["description"]
                    
                    if hasattr(Platform, 'created_at'):
                        platform.created_at = datetime.utcnow()
                    
                    db.session.add(platform)
                    added_count += 1
                    print(f" Added platform: {platform_info['name']}")
                else:
                    print(f" Platform already exists: {platform_info['name']}")
            
          db.session.commit()
            
          print(f" Database setup complete!")
          print(f"Added {added_count} new platforms")
            
          total_platforms = Platform.query.count()
          active_platforms = Platform.query.filter_by(is_active=True).count()
            
          print(f" Total platforms in database: {total_platforms}")
          print(f" Active platforms: {active_platforms}")
            
          return True
            
        except Exception as e:
            print(f" Database setup failed: {e}")
            db.session.rollback()
            return False

def verify_setup():
    print(" Verifying database setup...")
    
    with app.app_context():
        try:
            youtube = Platform.query.filter_by(name="YouTube").first()
            if youtube and youtube.is_active:
                print(" YouTube platform is ready for data collection")
                return True
            else:
                print("YouTube platform is not properly configured")
                return False
                
        except Exception as e:
            print(f" Verification failed: {e}")
            return False

if __name__ == "__main__":
    print(" Starting database setup...")
    print("=" * 50)
    
    if setup_database():
        # Verify setup
        if verify_setup():
            print(" SUCCESS: Your database is ready!")
        else:
            print(" Setup verification failed")
            sys.exit(1)
    else:
        print("Database setup failed")
        sys.exit(1)