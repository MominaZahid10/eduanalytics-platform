import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import warnings
import sys
import os
from contextlib import contextmanager
warnings.filterwarnings('ignore')

try:
    from features import (
        create_learning_dashboard_data,
        generate_learning_recommendations,
        LearningIntelligenceEngine,
        get_learning_velocity,
        dashboard_data
    )
    FEATURES_AVAILABLE = True
except Exception as e:
    import traceback
    FEATURES_AVAILABLE = False
    import streamlit as st
    st.error(f"❌ Error importing features module: {e}")
    st.text(traceback.format_exc())

st.set_page_config(
    page_title="EduAnalytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: #fafafa;
        min-height: 100vh;
        padding: 0;
    }
    
    /* Header Styles */
    .dashboard-header {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-size: 1.3rem;
    }
    
    .dashboard-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
    }
    
    .dashboard-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
    }
    
    /* Metric Cards, Section Containers, Advanced Cards, Recommendation Cards */
    .metric-card, .section-container, .advanced-card, .recommendation-card {
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(99,102,241,0.07);
        background: white;
        margin-bottom: 1.1rem;
        padding: 0.7rem 0.8rem;
        transition: box-shadow 0.2s, transform 0.2s;
        min-width: 0;
        flex: 1 1 0%;
    }
    .metric-card:hover, .advanced-card:hover, .recommendation-card:hover {
        box-shadow: 0 8px 32px rgba(99,102,241,0.13);
        transform: translateY(-2px) scale(1.02);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        color: #4f46e5;
        letter-spacing: -0.01em;
    }
    .insight-pill {
        background: linear-gradient(90deg, #e0e7ff 0%, #f0f9ff 100%);
        border-radius: 20px;
        padding: 0.7rem 1.2rem;
        font-size: 1rem;
        color: #3730a3;
        margin: 1.2rem 0;
        display: inline-block;
        font-weight: 600;
    }
    /* Responsive grid for metrics and recommendations */
    .metric-grid, .advanced-grid, .rec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.7rem;
        margin-bottom: 2rem;
    }
    /* Chart Containers */
    .chart-container {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    /* Recommendation Card */
    .recommendation-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        margin-top: 2rem;
        color: #0c4a6e;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .recommendation-card .rec-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .recommendation-card .rec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.5rem;
    }
    .recommendation-card .rec-item {
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .recommendation-card .rec-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #0369a1;
        letter-spacing: 0.05em;
        margin-bottom: 0.12rem;
    }
    .recommendation-card .rec-value {
        font-size: 1rem;
        font-weight: 700;
        color: #0c4a6e;
    }
    .recommendation-card .rec-desc {
        font-size: 0.75rem;
        color: #164e63;
    }
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .sidebar-metric {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #4f46e5;
    }
    .sidebar-metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.18rem;
    }
    .sidebar-metric-label {
        font-size: 0.65rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Advanced Section Styles */
    .advanced-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .advanced-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .advanced-card:hover {
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .advanced-card h3 {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .advanced-card p {
        font-size: 0.875rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .stat-highlight {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #0ea5e9;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0c4a6e;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #0369a1;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
    }
    
    .analysis-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .analysis-description {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid #6366f1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .table-container {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .no-data {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-style: italic;
    }
    
    .error-msg {
        background: #fee2e2;
        color: #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Clean Streamlit elements */
    .stApp > header {
        background: transparent;
    }
    
    .stApp > div > div > div > div > section > div {
        padding-top: 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Responsive design for mobile */
    @media (max-width: 900px) {
        .metric-grid, .advanced-grid, .rec-grid {
            grid-template-columns: 1fr !important;
            gap: 0.5rem !important;
        }
        .section-container, .advanced-card, .recommendation-card, .metric-card {
            padding: 0.5rem 0.3rem !important;
        }
        .section-title {
            font-size: 1rem !important;
        }
        .dashboard-header {
            padding: 0.5rem !important;
            font-size: 1rem !important;
        }
    }
    @media (max-width: 600px) {
        .dashboard-header {
            font-size: 0.9rem !important;
            padding: 0.2rem !important;
        }
        .section-title {
            font-size: 0.8rem !important;
        }
        .metric-card, .section-container, .advanced-card, .recommendation-card {
            padding: 0.2rem 0.05rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@contextmanager
def suppress_stdout():
    """Safely suppress stdout with proper encoding handling"""
    try:
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            yield
    except:
        yield
    finally:
        if 'old_stdout' in locals():
            sys.stdout = old_stdout

st.markdown("""
<div class="dashboard-header">
    <h1>📊 EduAnalytics Pro</h1>
    <p>Advanced Learning Intelligence & Performance Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

def standardize_column_names(df):
    """Standardize column names for consistent access"""
    column_mapping = {}
    important_columns = [
        'engagement_score', 'rating', 'views', 'likes', 'comments', 
        'enrollments', 'upvotes', 'upvote_ratio', 'comments_per_minute',
        'likes_per_minute', 'educational_score', 'complexity_score',
        'discussion_engagement', 'engagement_rate', 'gilded',
        'engagement_score_normalized', 'engagement_category', 'course_title'
    ]
    for col in important_columns:
        if f"{col}_course" in df.columns and f"{col}_engagement" in df.columns:
            if col.startswith('engagement') or col in ['rating', 'views', 'likes', 'comments']:
                column_mapping[col] = f"{col}_engagement"
            else:
                column_mapping[col] = f"{col}_course"
        elif f"{col}_course" in df.columns:
            column_mapping[col] = f"{col}_course"
        elif f"{col}_engagement" in df.columns:
            column_mapping[col] = f"{col}_engagement"
        elif col in df.columns:
            column_mapping[col] = col
    
    for standard_name, actual_name in column_mapping.items():
        if actual_name in df.columns and standard_name not in df.columns:
            df[standard_name] = df[actual_name]
    
    return df

def add_derived_columns(df):
    """Add derived columns including robust learning efficiency calculation"""
    if 'completion_rate' in df.columns and 'duration_minutes' in df.columns:
        df['learning_efficiency'] = (
            df['completion_rate'].fillna(0) /
            (df['duration_minutes'].fillna(60) / 60).clip(lower=0.1)
        )
    elif 'engagement_score' in df.columns and 'educational_score' in df.columns:
        df['learning_efficiency'] = (
            df['engagement_score'].fillna(0) * df['educational_score'].fillna(0)
        )
    elif 'engagement_score' in df.columns:
        df['learning_efficiency'] = df['engagement_score'].fillna(0) / 100
    else:
        df['learning_efficiency'] = np.random.uniform(0.001, 0.01, len(df))
    
    if 'rating' in df.columns and 'engagement_score' in df.columns:
        df['quality_index'] = (df['rating'] * 0.4 + df['engagement_score'] * 0.6)
    
    if 'views' in df.columns and 'likes' in df.columns:
        df['popularity_score'] = np.log1p(df['views']) + np.log1p(df['likes'])
    
    if 'price' in df.columns and 'rating' in df.columns:
        df['roi_score'] = df['rating'] / (df['price'] + 1)
    
    if 'engagement_score' in df.columns:
        df['engagement_category'] = pd.cut(df['engagement_score'], 
                                         bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                         labels=['Low', 'Medium', 'High', 'Excellent'])
    
    return df

def create_advanced_fallback_data():
    """Create sophisticated fallback data with statistical properties"""
    np.random.seed(42)
    
    subjects = ['Data Science', 'Machine Learning', 'Programming', 'Statistics', 'Business Analytics', 'AI/ML']
    difficulties = ['Beginner', 'Intermediate', 'Advanced']
    platforms = ['YouTube', 'Coursera', 'Udemy', 'edX']
    instructors = ['Dr. Sarah Chen', 'Prof. Michael Rodriguez', 'Dr. Emily Johnson', 'Prof. David Kim', 'Dr. Lisa Wang']
    
    n_courses = 350
    base_quality = np.random.normal(0.7, 0.2, n_courses)
    base_engagement = np.clip(base_quality + np.random.normal(0, 0.1, n_courses), 0, 1)
    
    data = {
        'course_id': [f'course_{i:04d}' for i in range(n_courses)],
        'title': [f'{np.random.choice(subjects)} - {np.random.choice(["Fundamentals", "Advanced", "Masterclass", "Bootcamp"])}' for _ in range(n_courses)],
        'instructor': np.random.choice(instructors, n_courses),
        'category': np.random.choice(subjects, n_courses),
        'difficulty_level': np.random.choice(difficulties, n_courses),
        'platform': np.random.choice(platforms, n_courses),
        'duration_minutes': np.random.lognormal(4, 0.5, n_courses).astype(int),
        'rating': np.clip(base_quality * 5, 1, 5),
        'views': np.random.lognormal(6, 1.5, n_courses).astype(int),
        'likes': np.random.lognormal(4, 1, n_courses).astype(int),
        'comments': np.random.lognormal(3, 1, n_courses).astype(int),
        'enrollments': np.random.lognormal(5, 1, n_courses).astype(int),
        'completion_rate': np.clip(base_engagement, 0.1, 0.95),
        'engagement_score': base_engagement,
        'price': np.random.choice([0, 29, 49, 99, 199], n_courses, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
        'is_free': np.random.choice([True, False], n_courses, p=[0.3, 0.7]),
        'published_at': pd.date_range('2023-01-01', periods=n_courses, freq='3H'),
        'last_updated': pd.date_range('2023-06-01', periods=n_courses, freq='5H'),
        'dropout_time': np.random.exponential(30, n_courses),
        'complexity_score': np.random.beta(2, 3, n_courses),
    }
    return pd.DataFrame(data)

# Remove the problematic @st.cache_data decorator and simplify the function
def load_advanced_data():
    """Load data with preprocessing and enhanced derived columns"""
    with st.spinner("Loading data..."):
        try:
            if FEATURES_AVAILABLE:
                engine = LearningIntelligenceEngine()
                with suppress_stdout():
                    enhanced_df, completion_df = engine.load_enhanced_data()
                if enhanced_df is not None and completion_df is not None:
                    df = completion_df.copy()
                else:
                    st.error("❌ load_enhanced_data() returned None. There was a problem loading from the database. Check your schema and data.")
                    df = create_advanced_fallback_data()
                    st.info("ℹ️ Using synthetic data for demonstration")
            else:
                st.error("❌ FEATURES_AVAILABLE is False. The features module could not be imported.")
                df = create_advanced_fallback_data()
                st.info("ℹ️ Using synthetic data (features module not available)")
        
        except Exception as e:
            import traceback
            st.error(f"⚠️ Error loading data: {e}")
            st.text(traceback.format_exc())
            df = create_advanced_fallback_data()
            st.info("ℹ️ Switched to fallback data")
        
        df = standardize_column_names(df)
        df = add_derived_columns(df)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        return df

# Initialize data without caching to avoid tokenization issues
if 'df' not in st.session_state:
    st.session_state.df = load_advanced_data()

df = st.session_state.df

st.sidebar.markdown("### Navigation")
user_type = st.sidebar.selectbox(
    "Select Dashboard View:",
    ["🎓 Learner Intelligence", "📊 Creator Analytics", "🔬 Advanced Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Key Performance Indicators")

total_courses = len(df)
avg_completion = df['completion_rate'].mean()
avg_rating = df['rating'].mean()
top_platform = df['platform'].value_counts().index[0]
learning_efficiency = df['learning_efficiency'].mean()

st.sidebar.markdown(f"""
<div class="sidebar-metric">
    <div class="sidebar-metric-value">{total_courses:,}</div>
    <div class="sidebar-metric-label">Total Courses</div>
</div>
<div class="sidebar-metric">
    <div class="sidebar-metric-value">{avg_completion:.1%}</div>
    <div class="sidebar-metric-label">Completion Rate</div>
</div>
<div class="sidebar-metric">
    <div class="sidebar-metric-value">{avg_rating:.1f}★</div>
    <div class="sidebar-metric-label">Average Rating</div>
</div>
<div class="sidebar-metric">
    <div class="sidebar-metric-value">{learning_efficiency:.3f}</div>
    <div class="sidebar-metric-label">Learning Efficiency</div>
</div>
""", unsafe_allow_html=True)

def explanation_card(title, text):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0 2.5rem 0;
        color: #0c4a6e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">{title}</div>
        <div style="font-size:0.95rem;line-height:1.6;">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(title, icon=""):
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(99,102,241,0.08);
        font-size: 1.6rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <span style="font-size:2rem;">{icon}</span> {title}
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, sublabel="", icon="", color="#6366f1"):
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(99,102,241,0.07);
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.7rem;
        border-left: 6px solid {color};
        display: flex;
        align-items: center;
        gap: 0.7rem;
        min-width: 0;
        flex: 1 1 0%;
        transition: box-shadow 0.2s;
    ">
        <span style="font-size:1.3rem; color:{color}; min-width:1.3rem;">{icon}</span>
        <div style="min-width:0;">
            <div style="font-size:1.15rem; font-weight:700; color:#1e293b; letter-spacing:-0.01em; font-family:'Inter',sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {value}
            </div>
            <div style="font-size:0.92rem; font-weight:600; color:#4f46e5; margin-top:0.05rem; font-family:'Inter',sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {label}
            </div>
            <div style="font-size:0.8rem; color:#64748b; font-weight:400; margin-top:0.08rem; font-family:'Inter',sans-serif; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {sublabel}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if user_type == "🎓 Learner Intelligence":
    
    optimal_duration = df[df['completion_rate'] > df['completion_rate'].quantile(0.8)]['duration_minutes'].median()
    best_difficulty = df.groupby('difficulty_level')['completion_rate'].mean().idxmax()
    best_completion = df.groupby('difficulty_level')['completion_rate'].mean().max()
    best_platform = df.groupby('platform')['completion_rate'].mean().idxmax()
    platform_completion = df.groupby('platform')['completion_rate'].mean().max()
    
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Optimal Duration", f"{optimal_duration:.0f}min", "For maximum retention", "⏱️", "#6366f1")
    
    with col2:
        metric_card("Best Success Rate", f"{best_completion:.1%}", f"{best_difficulty} difficulty", "🏆", "#10b981")
    
    with col3:
        metric_card("Top Platform", f"{platform_completion:.1%}", f"{best_platform}", "🎓", "#f59e42")
    
    with col4:
        learning_velocity = df['learning_efficiency'].quantile(0.75)
        metric_card("Learning Velocity", f"{learning_velocity:.3f}", "Top 25% efficiency", "⚡", "#6366f1")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    df_clean = df.copy()
    
    category_mapping = {
        'python': 'Programming',
        'python programming beginner': 'Programming',
        'programming': 'Programming',
        'computer_science': 'Programming',
        'machine learning': 'Data Science',
        'machine learning explained': 'Data Science',
        'data science': 'Data Science',
        'data_science': 'Data Science',
        'artificial intelligence': 'AI/ML',
        'deep learning': 'AI/ML',
        'web_development': 'Web Development',
        'web development full course': 'Web Development',
        'sql database complete tutorial': 'Database',
        'excel advanced tutorial': 'Business Skills',
        'digital marketing complete course': 'Business Skills',
        'photoshop complete tutorial': 'Design',
        'project management fundamentals': 'Business Skills',
        'statistics explained simply': 'Data Science',
        'calculus step by step': 'Mathematics',
        'general': 'General'
    }
    
    df_clean['category_clean'] = df_clean['category'].map(category_mapping).fillna('Other')
    
    difficulty_mapping = {
        'Beginner · Course · 1 - 3 Months': 'Beginner',
        'Intermediate · Course · 1 - 3 Months': 'Intermediate', 
        'Advanced · Course · 1 - 3 Months': 'Advanced',
        'Beginner · Course · 1 - 4 Weeks': 'Beginner',
        'Mixed · Course · 1 - 4 Weeks': 'Mixed',
        'Intermediate · Course · 1 - 4 Weeks': 'Intermediate',
        'Beginner': 'Beginner',
        'Intermediate': 'Intermediate',
        'Advanced': 'Advanced'
    }
    
    df_clean['difficulty_clean'] = df_clean['difficulty_level'].map(difficulty_mapping).fillna('Other')
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Learning Efficiency Analysis</div>', unsafe_allow_html=True)
    
    fig = px.box(
        df_clean, 
        x='difficulty_clean', 
        y='learning_efficiency',
        color='category_clean',
        title="",
        labels={
            'learning_efficiency': 'Learning Efficiency Score',
            'difficulty_clean': 'Difficulty Level',
            'category_clean': 'Course Category'
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title="",
            font=dict(size=11)
        ),
        xaxis=dict(
            title_font=dict(size=12),
            tickfont=dict(size=11),
            showgrid=False
        ),
        yaxis=dict(
            title_font=dict(size=12),
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    fig.update_traces(
        marker=dict(size=4),
        line=dict(width=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    best_category = df_clean.groupby('category_clean')['learning_efficiency'].mean().idxmax()
    best_category_score = df_clean.groupby('category_clean')['learning_efficiency'].mean().max()
    
    st.markdown(f"""
    <div class="insight-pill">
        <strong>{best_category}</strong> courses show highest efficiency ({best_category_score:.3f}) • <strong>{best_difficulty}</strong> level has best completion rates ({best_completion:.1%})
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Optimal Learning Paths</div>', unsafe_allow_html=True)
    
    path_data = df_clean.groupby(['category_clean', 'difficulty_clean']).agg({
        'completion_rate': 'mean',
        'rating': 'mean',
        'duration_minutes': 'mean',
        'learning_efficiency': 'mean'
    }).reset_index()
    
    path_data['path_score'] = (
        path_data['completion_rate'] * 0.4 +
        path_data['rating'] / 5 * 0.3 +
        path_data['learning_efficiency'] * 0.3
    )
    
    top_paths = path_data.nlargest(8, 'path_score')
    
    fig = px.scatter(
        top_paths,
        x='completion_rate',
        y='learning_efficiency',
        size='path_score',
        color='difficulty_clean',
        hover_data=['category_clean', 'rating'],
        title="",
        labels={
            'completion_rate': 'Completion Rate',
            'learning_efficiency': 'Learning Efficiency',
            'difficulty_clean': 'Difficulty Level'
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        height=350,
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="recommendation-card">
        <div class="rec-title">🎯 Your Learning Strategy</div>
        <div class="rec-grid">
            <div class="rec-item">
                <div class="rec-label">Start Level</div>
                <div class="rec-value">{best_difficulty}</div>
                <div class="rec-desc">{best_completion:.1%} success rate</div>
            </div>
            <div class="rec-item">
                <div class="rec-label">Session Time</div>
                <div class="rec-value">{optimal_duration:.0f}min</div>
                <div class="rec-desc">Optimal retention</div>
            </div>
            <div class="rec-item">
                <div class="rec-label">Best Platform</div>
                <div class="rec-value">{best_platform}</div>
                <div class="rec-desc">Highest satisfaction</div>
            </div>
            <div class="rec-item">
                <div class="rec-label">Focus Area</div>
                <div class="rec-value">{best_category}</div>
                <div class="rec-desc">Top efficiency</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif user_type == "📊 Creator Analytics":
    st.markdown("## 📊 Creator Intelligence Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_engagement = df['engagement_score'].mean()
        metric_card("Avg Engagement", f"{avg_engagement:.1%}", "Across all content", "📈", "#6366f1")
    
    with col2:
        high_performers = (df['completion_rate'] > df['completion_rate'].quantile(0.8)).sum()
        metric_card("High Performers", f"{high_performers}", "Top 20% completion", "⭐", "#10b981")
    
    with col3:
        total_revenue = (df['price'] * df['enrollments']).sum()
        metric_card("Total Revenue", f"${total_revenue:,.0f}", "Estimated", "💰", "#f59e42")
    
    with col4:
        avg_roi = df['roi_score'].mean()
        metric_card("ROI Score", f"{avg_roi:.2f}", "Quality/Price ratio", "📊", "#6366f1")

    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Content Performance Matrix</div>', unsafe_allow_html=True)

    fig = px.scatter(df, 
                    x='duration_minutes', 
                    y='completion_rate',
                    size='enrollments',
                    color='price',
                    hover_data=['rating', 'engagement_score'],
                    title="",
                    labels={
                        'duration_minutes': 'Duration (minutes)',
                        'completion_rate': 'Completion Rate',
                        'price': 'Price ($)'
                    },
                    color_continuous_scale='viridis')
    
    fig.update_layout(
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-pill">
        Shorter courses (20-60min) typically achieve higher completion rates and better learner engagement
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Market Opportunities</div>', unsafe_allow_html=True)

        market_data = df.groupby('category').agg({
            'enrollments': 'sum',
            'course_id': 'count'
        }).rename(columns={'course_id': 'supply'})
        market_data['demand_per_course'] = market_data['enrollments'] / market_data['supply']

        fig = px.scatter(market_data, 
                        x='supply', 
                        y='enrollments',
                        size='demand_per_course',
                        text=market_data.index,
                        title="",
                        color_discrete_sequence=['#2E86AB'])
        
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.update_layout(
            height=350,
            font=dict(family="Inter, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 Pricing Strategy</div>', unsafe_allow_html=True)

        price_analysis = df.groupby('price').agg({
            'completion_rate': 'mean',
            'enrollments': 'mean',
            'rating': 'mean'
        }).reset_index()

        fig = px.line(price_analysis, 
                     x='price', 
                     y='completion_rate',
                     title="",
                     markers=True,
                     color_discrete_sequence=['#A23B72'])
        
        fig.update_layout(
            height=350,
            font=dict(family="Inter, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    optimal_duration = df[df['completion_rate'] > df['completion_rate'].quantile(0.8)]['duration_minutes'].median()
    best_platform = df.groupby('platform')['completion_rate'].mean().idxmax()
    optimal_price = price_analysis.loc[price_analysis['completion_rate'].idxmax(), 'price']
    best_category = market_data['demand_per_course'].idxmax()

    st.markdown(f"""
    <div class="recommendation-card">
        <div class="rec-title">💡 Creator Strategy</div>
        <div class="rec-grid">
            <div class="rec-item">
                <div class="rec-label">Sweet Spot Price</div>
                <div class="rec-value">${optimal_price}</div>
                <div class="rec-desc">Optimal completion</div>
            </div>
            <div class="rec-item">
                <div class="rec-label">High Demand</div>
                <div class="rec-value">{best_category}</div>
                <div class="rec-desc">{market_data.loc[best_category, 'demand_per_course']:.0f} enroll/course</div>
            </div>
            <div class="rec-item">
                <div class="rec-label">Best Length</div>
                <div class="rec-value">{optimal_duration:.0f}min</div>
                <div class="rec-desc">Top performers</div>
            </div>
            <div class="rec-item">
                <div class="rec-label">Platform Focus</div>
                <div class="rec-value">{best_platform}</div>
                <div class="rec-desc">Highest success</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif user_type == "🔬 Advanced Insights":
    
    st.markdown('<div class="advanced-container">', unsafe_allow_html=True)
    st.markdown('<div class="advanced-title">🔬 Statistical Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Survival Analysis", "📊 Platform Comparison", "⏱️ Duration Analysis", "🔗 Association Tests"])
    
    with tab1:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="analysis-description">
            <strong>Course Completion Survival Analysis</strong><br>
            Analyzes the probability of learners completing courses over time and identifies critical drop-off points.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            velocity_data = get_learning_velocity()
            if velocity_data and velocity_data.get('median_completion_time') != 'N/A':
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{velocity_data.get('median_completion_time')} days</div>
                        <div class="stat-label">Median Completion Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{velocity_data.get('completion_rate_30_days', 0):.1%}</div>
                        <div class="stat-label">30-Day Success Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{velocity_data.get('completion_rate_60_days', 0):.1%}</div>
                        <div class="stat-label">60-Day Success Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                platform_comparison = velocity_data.get('platform_comparison', {})
                if platform_comparison:
                    survival_data = []
                    for platform, data in platform_comparison.items():
                        survival_data.append({
                            'Platform': platform,
                            'Median Days': data.get('median_survival_time', 'N/A'),
                            '30-Day Rate': f"{data.get('survival_at_30_days', 0):.1%}",
                            'Avg Completion': f"{data.get('average_completion_rate', 0):.1%}"
                        })
                    
                    survival_df = pd.DataFrame(survival_data)
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.dataframe(survival_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-data">Survival analysis data not available</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-msg">Analysis error: {str(e)}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="analysis-description">
            <strong>Platform Engagement Analysis (ANOVA)</strong><br>
            Tests whether engagement levels differ significantly across learning platforms.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            platform_groups = [group['engagement_score'].values for _, group in df.groupby('platform')]
            f_stat, p_val = stats.f_oneway(*platform_groups)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{p_val:.4f}</div>
                    <div class="stat-label">P-value</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{significance}</div>
                    <div class="stat-label">Result</div>
                </div>
                """, unsafe_allow_html=True)
            
            fig = px.box(df, x='platform', y='engagement_score', 
                        title="", color_discrete_sequence=['#2E86AB'])
            fig.update_layout(
                height=300,
                font=dict(family="Inter, sans-serif", size=11),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, title="Platform"),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', title="Engagement Score")
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.markdown(f'<div class="error-msg">Analysis error: {str(e)}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="analysis-description">
            <strong>Duration by Difficulty Analysis (Kruskal-Wallis)</strong><br>
            Non-parametric test examining if course duration varies significantly across difficulty levels.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            difficulty_groups = [group['duration_minutes'].values for _, group in df.groupby('difficulty_level')]
            h_stat, p_val = stats.kruskal(*difficulty_groups)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{p_val:.4f}</div>
                    <div class="stat-label">P-value</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{significance}</div>
                    <div class="stat-label">Result</div>
                </div>
                """, unsafe_allow_html=True)
            
            fig = px.box(df, x='difficulty_level', y='duration_minutes', 
                        title="", color_discrete_sequence=['#A23B72'])
            fig.update_layout(
                height=300,
                font=dict(family="Inter, sans-serif", size=11),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis=dict(showgrid=False, title="Difficulty Level", tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', title="Duration (minutes)")
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.markdown(f'<div class="error-msg">Analysis error: {str(e)}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="analysis-description">
            <strong>Difficulty-Platform Association (Chi-Square)</strong><br>
            Tests whether course difficulty distribution depends on the platform offering the course.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            contingency_table = pd.crosstab(df['difficulty_level'], df['platform'])
            chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{p_val:.4f}</div>
                    <div class="stat-label">P-value</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{significance}</div>
                    <div class="stat-label">Association</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="table-container">', unsafe_allow_html=True)
            st.markdown("**Difficulty × Platform Distribution:**")
            st.dataframe(contingency_table, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="error-msg">Analysis error: {str(e)}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
