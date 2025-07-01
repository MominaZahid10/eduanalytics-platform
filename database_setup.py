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
                    "name": "Udemy",
                    "base_url": "https://www.udemy.com",
                    "is_active": False,
                    "description": "Marketplace for learning and teaching online"
                },
                {
                    "name": "edX",
                    "base_url": "https://www.edx.org",
                    "is_active": False,
                    "description": "High-quality courses from the world's best universities"
                },
                {
                    "name": "Khan Academy",
                    "base_url": "https://www.khanacademy.org",
                    "is_active": False,
                    "description": "Free online courses, lessons and practice"
                }
          ]
          added_count = 0
          for platform_info in platforms_data:
                # Check if platform already exists
                existing = Platform.query.filter_by(name=platform_info["name"]).first()
                
                if not existing:
                    platform = Platform(
                        name=platform_info["name"],
                        base_url=platform_info["base_url"],
                        is_active=platform_info["is_active"]
                    )
                    
                    # Add description if your Platform model has this field
                    if hasattr(Platform, 'description'):
                        platform.description = platform_info["description"]
                    
                    # Add created_at if your Platform model has this field
                    if hasattr(Platform, 'created_at'):
                        platform.created_at = datetime.utcnow()
                    
                    db.session.add(platform)
                    added_count += 1
                    print(f" Added platform: {platform_info['name']}")
                else:
                    print(f" Platform already exists: {platform_info['name']}")
            
            # Commit all changes
          db.session.commit()
            
          print(f" Database setup complete!")
          print(f"Added {added_count} new platforms")
            
            # Show summary
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
    """Verify that the database setup was successful"""
    print(" Verifying database setup...")
    
    with app.app_context():
        try:
            # Check if YouTube platform exists and is active
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
    
    # Setup database
    if setup_database():
        # Verify setup
        if verify_setup():
            print(" SUCCESS: Your database is ready!")
            print(" Next steps:")
            print("1. Make sure you have YOUTUBE_API_KEY in your .env file")
            print("2. Run: python youtube_data_collector.py")
            print("3. Start your Flask app to see the collected courses")
        else:
            print(" Setup verification failed")
            sys.exit(1)
    else:
        print("Database setup failed")
        sys.exit(1)