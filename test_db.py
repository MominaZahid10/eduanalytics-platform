import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect

st.title('🔌 Database Connection Test')

# Get database URL from Streamlit secrets
db_url = st.secrets.get('DATABASE_URL')

if not db_url:
    st.error('DATABASE_URL not found in Streamlit secrets!')
    st.stop()

st.write(f"**Database URL:** {db_url[:40]}... (hidden for security)")

try:
    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    st.success(f"Connected to database! Found {len(tables)} tables.")
    st.write('**Tables:**', tables)

    for table in tables:
        st.write(f'---\n### Table: `{table}`')
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 5', engine)
            st.dataframe(df)
        except Exception as e:
            st.error(f'Error reading table `{table}`: {e}')
except Exception as e:
    st.error(f'Failed to connect or inspect database: {e}') 