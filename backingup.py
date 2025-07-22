import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import sys

def migrate_table_data():
    try:
        # Create SQLAlchemy engines (this fixes the pandas issue)
        local_engine = create_engine("postgresql://postgres:pgadmin4@localhost:5432/eduanalytics")
        supabase_engine = create_engine("postgresql://postgres.eivvjtpbpmaikxckcrzp:pgadmin4@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres")
        
        # Test connections
        print("Testing connections...")
        local_conn = local_engine.connect()
        supabase_conn = supabase_engine.connect()
        print("✅ Both connections successful!")
        
        # Get all tables
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"
        tables_df = pd.read_sql_query(tables_query, local_engine)
        tables = tables_df['table_name'].tolist()
        
        print(f"\nFound {len(tables)} tables to migrate:")
        for table in tables:
            print(f"  - {table}")
        
        # Migrate each table
        success_count = 0
        for table in tables:
            print(f"\nMigrating table: {table}")
            
            try:
                # Read data from local
                df = pd.read_sql_query(f"SELECT * FROM {table}", local_engine)
                print(f"  Read {len(df)} rows from local database")
                
                if len(df) > 0:
                    # Write to Supabase
                    df.to_sql(table, supabase_engine, if_exists='replace', index=False)
                    print(f"  ✅ Successfully migrated {len(df)} rows to Supabase")
                    success_count += 1
                else:
                    print(f"  ⚠️ Table {table} is empty, skipping")
                
            except Exception as e:
                print(f"  ❌ Error migrating {table}: {e}")
                continue
        
        print(f"\n🎉 Migration completed! Successfully migrated {success_count}/{len(tables)} tables")
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        # Close connections
        try:
            local_conn.close()
            supabase_conn.close()
        except:
            pass

if __name__ == "__main__":
    migrate_table_data()