from mainapp import app, Course, EngagementMetric, db
from datacleaning import AdvancedEduDataQuality, run_comprehensive_enhancement
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sqlalchemy import text, create_engine
import logging
import warnings
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict, Counter
from flask import current_app
import json
from contextlib import contextmanager

warnings.filterwarnings('ignore')

class LearningIntelligenceEngine:
    def __init__(self, db_uri=None):
        """Initialize the Learning Intelligence Engine with Flask app context support"""
        self.db_uri = db_uri
        self.db_engine = None
        self.courses_df = None
        self.engagement_df = None
        self.enhanced_df = None
        self.completion_df = None
        self._app_context = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Platform-specific configurations
        self.platform_metrics = {
            'YouTube': {
                'primary_engagement': 'views',
                'secondary_engagement': 'likes',
                'engagement_divisor': 1000,
                'completion_base': 0.15,
                'rating_source': 'likes_ratio'
            },
            'Coursera': {
                'primary_engagement': 'enrollments',
                'secondary_engagement': 'rating',
                'engagement_divisor': 1,
                'completion_base': 0.70,
                'rating_source': 'direct_rating'
            },
            'Khan Academy': {
                'primary_engagement': 'completion_rate',
                'secondary_engagement': 'rating',
                'engagement_divisor': 1,
                'completion_base': 0.65,
                'rating_source': 'educational_quality'
            },
            'Reddit': {
                'primary_engagement': 'upvotes',
                'secondary_engagement': 'comments',
                'engagement_divisor': 10,
                'completion_base': 0.25,
                'rating_source': 'upvote_ratio'
            }
        }
        
        # Initialize database connection
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection with Flask app context"""
        try:
            if self.db_uri:
                self.db_engine = create_engine(self.db_uri)
                self.logger.info("Database engine initialized with provided URI")
            else:
                # Try to get from Flask app context
                self._ensure_app_context()
                if current_app:
                    self.db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI')
                    if self.db_uri:
                        self.db_engine = create_engine(self.db_uri)
                        self.logger.info("Database engine initialized from Flask app context")
                    else:
                        raise ValueError("No database URI found in Flask app configuration")
                else:
                    raise ValueError("No Flask app context available and no db_uri provided")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _ensure_app_context(self):
        """Ensure Flask app context is available"""
        try:
            # Check if we're already in an app context
            if current_app._get_current_object():
                return
        except RuntimeError:
            # Not in app context, create one
            self._app_context = app.app_context()
            self._app_context.push()
            self.logger.info("Flask app context created and pushed")
    
    def __enter__(self):
        """Context manager entry"""
        self._ensure_app_context()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._app_context:
            self._app_context.pop()
            self._app_context = None
    
    @contextmanager
    def app_context(self):
        """Context manager for Flask app context"""
        context_created = False
        try:
            current_app._get_current_object()
        except RuntimeError:
            self._app_context = app.app_context()
            self._app_context.push()
            context_created = True
        
        try:
            yield
        finally:
            if context_created and self._app_context:
                self._app_context.pop()
                self._app_context = None

    def normalize_engagement_scores(self, df):
        """Normalize engagement scores across platforms for fair comparison"""
        normalized_df = df.copy()
        
        for platform in df['platform'].unique():
            if pd.isna(platform):
                continue
            
            platform_mask = df['platform'] == platform
            platform_data = df[platform_mask]
            
            if len(platform_data) == 0:
                continue
            
            platform_config = self.platform_metrics.get(platform, self.platform_metrics['Coursera'])
            engagement_values = []
            
            for _, row in platform_data.iterrows():
                try:
                    primary_metric = platform_config['primary_engagement']
                    secondary_metric = platform_config['secondary_engagement']
                    divisor = platform_config['engagement_divisor']
                    
                    # Calculate primary engagement value
                    primary_value = 0
                    if primary_metric in row.index and pd.notna(row[primary_metric]):
                        primary_value = float(row[primary_metric]) / divisor
                    
                    # Calculate secondary engagement value
                    secondary_value = 0
                    if secondary_metric in row.index and pd.notna(row[secondary_metric]):
                        secondary_value = float(row[secondary_metric])
                    
                    # Platform-specific engagement calculation
                    if platform == 'YouTube':
                        engagement_score = np.log1p(primary_value) * 0.7 + np.log1p(secondary_value) * 0.3
                    elif platform == 'Khan Academy':
                        engagement_score = primary_value * 0.8 + (secondary_value / 5.0) * 0.2
                    elif platform == 'Reddit':
                        engagement_score = np.log1p(primary_value) * 0.6 + np.log1p(secondary_value) * 0.4
                    else:
                        engagement_score = np.log1p(primary_value) * 0.6 + (secondary_value / 5.0) * 0.4
                    
                    engagement_values.append(engagement_score)
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating engagement for {platform}: {e}")
                    engagement_values.append(0.5)  # Default value
            
            if engagement_values:
                engagement_array = np.array(engagement_values)
                if engagement_array.max() > engagement_array.min():
                    normalized_values = (engagement_array - engagement_array.min()) / (engagement_array.max() - engagement_array.min())
                else:
                    normalized_values = np.full_like(engagement_array, 0.5)
                
                # Scale to 0.1-0.9 range
                normalized_values = normalized_values * 0.8 + 0.1
                normalized_df.loc[platform_mask, 'engagement_score_normalized'] = normalized_values
        
        return normalized_df

    
    def calculate_completion_rate_enhanced(self, df):
        """Enhanced completion rate calculation with multiple strategies"""
        completion_rates = []
        
        for idx, row in df.iterrows():
            try:
                platform = row.get('platform', 'Unknown')
                platform_config = self.platform_metrics.get(platform, self.platform_metrics['Coursera'])
                
                # Check if completion rate already exists
                if 'completion_rate' in row.index and pd.notna(row['completion_rate']):
                    existing_rate = float(row['completion_rate'])
                    if 0 <= existing_rate <= 1:
                        completion_rates.append(existing_rate)
                        continue
                
                # Calculate base completion rate
                base_completion = platform_config['completion_base']
                
                # Engagement boost
                engagement_boost = 0
                if 'engagement_score_normalized' in row.index and pd.notna(row['engagement_score_normalized']):
                    engagement_score = float(row['engagement_score_normalized'])
                    engagement_boost = (engagement_score - 0.5) * 0.3
                
                # Rating boost
                rating_boost = 0
                if 'rating' in row.index and pd.notna(row['rating']):
                    rating = float(row['rating'])
                    if rating > 0:
                        rating_boost = ((rating - 3.0) / 2.0) * 0.15
                
                # Duration adjustment
                duration_adjustment = 0
                if 'duration_minutes' in row.index and pd.notna(row['duration_minutes']):
                    duration = float(row['duration_minutes'])
                    if platform == 'YouTube':
                        if duration <= 10:
                            duration_adjustment = 0.1
                        elif duration <= 30:
                            duration_adjustment = 0.05
                        elif duration > 60:
                            duration_adjustment = -0.1
                    elif platform == 'Khan Academy':
                        if 20 <= duration <= 60:
                            duration_adjustment = 0.05
                        elif duration > 120:
                            duration_adjustment = -0.05
                    else:
                        if duration > 300:  # 5+ hours
                            duration_adjustment = -0.15
                        elif duration > 180:  # 3+ hours
                            duration_adjustment = -0.05
                
                # Difficulty adjustment
                difficulty_adjustment = 0
                if 'difficulty_level' in row.index and pd.notna(row['difficulty_level']):
                    difficulty = row['difficulty_level']
                    if difficulty == 'Beginner':
                        difficulty_adjustment = 0.1
                    elif difficulty == 'Advanced':
                        difficulty_adjustment = -0.05
                
                # Price adjustment
                price_adjustment = 0
                if 'is_free' in row.index and pd.notna(row['is_free']):
                    if row['is_free']:
                        price_adjustment = -0.02  # Free courses often have lower completion
                    else:
                        price_adjustment = 0.05  # Paid courses have higher completion
                
                # Calculate final completion rate
                final_completion = (base_completion + engagement_boost + rating_boost + 
                                  duration_adjustment + difficulty_adjustment + price_adjustment)
                
                # Clip to reasonable range
                final_completion = np.clip(final_completion, 0.05, 0.95)
                completion_rates.append(final_completion)
                
            except Exception as e:
                self.logger.warning(f"Error calculating completion rate for row {idx}: {e}")
                completion_rates.append(0.5)  # Default fallback
        
        return np.array(completion_rates)

    def generate_khan_academy_engagement(self, courses_df):
        """Generate synthetic engagement for Khan Academy based on educational metrics"""
        khan_courses = courses_df[courses_df['platform'] == 'Khan Academy'].copy()
        
        if len(khan_courses) == 0:
            self.logger.info("No Khan Academy courses found for synthetic engagement generation")
            return pd.DataFrame()
        
        self.logger.info(f"Generating synthetic engagement for {len(khan_courses)} Khan Academy courses...")
        
        synthetic_engagement = []
        
        # Enhanced category engagement multipliers
        category_multipliers = {
            'Mathematics': 1.2,
            'Science': 1.15,
            'Programming': 1.3,
            'Computer Science': 1.25,
            'Physics': 1.1,
            'Chemistry': 1.05,
            'Biology': 1.1,
            'Statistics': 1.0,
            'Calculus': 1.15,
            'Linear Algebra': 1.0,
            'Data Science': 1.2,
            'Machine Learning': 1.15,
            'Web Development': 1.1
        }
        
        for _, course in khan_courses.iterrows():
            # Base engagement for Khan Academy (high educational quality)
            base_engagement = 0.75
            
            # Course characteristics
            duration_minutes = course.get('duration_minutes', 60)
            difficulty = course.get('difficulty_level', 'Intermediate')
            category = course.get('category', 'General')
            
            # Duration factor (optimal around 30-90 minutes for Khan Academy)
            if 20 <= duration_minutes <= 60:
                duration_factor = 0.15
            elif 60 < duration_minutes <= 90:
                duration_factor = 0.10
            elif duration_minutes <= 20:
                duration_factor = 0.05
            else:
                duration_factor = -0.05
            
            # Category factor
            category_factor = (category_multipliers.get(category, 1.0) - 1.0) * 0.8
            
            # Difficulty factor
            difficulty_factors = {
                'Beginner': 0.12,
                'Intermediate': 0.05,
                'Advanced': -0.03
            }
            difficulty_factor = difficulty_factors.get(difficulty, 0.05)
            
            # Calculate final engagement
            final_engagement = min(0.95, max(0.3, 
                base_engagement + duration_factor + category_factor + difficulty_factor))
            
            # Generate realistic completion rate
            completion_rate = self._calculate_khan_completion_rate(
                final_engagement, difficulty, duration_minutes
            )
            
            # Generate Khan Academy specific metrics
            engagement_data = {
                'course_id': course['course_id'],
                'platform': 'Khan Academy',
                'engagement_score': final_engagement,
                'engagement_score_normalized': final_engagement,
                'rating': min(5.0, 4.0 + (final_engagement * 0.9)),
                'views': int(duration_minutes * 150 + 1000),
                'likes': 0,
                'comments': 0,
                'enrollments': int(duration_minutes * 80 + 800),
                'completion_rate': completion_rate,
                'upvotes': 0,
                'upvote_ratio': 0,
                'comments_per_minute': 0,
                'likes_per_minute': 0,
                'engagement_rate': final_engagement,
                'gilded': 0,
                'engagement_category': self._categorize_engagement(final_engagement),
                'educational_score': min(1.0, final_engagement + 0.15),
                'complexity_score': {'Beginner': 0.3, 'Intermediate': 0.6, 'Advanced': 0.9}.get(difficulty, 0.6),
                'discussion_engagement': 0.2,
                'is_synthetic': True,
                'synthetic_reason': 'Educational platform - estimated metrics'
            }
            
            synthetic_engagement.append(engagement_data)
        
        synthetic_df = pd.DataFrame(synthetic_engagement)
        self.logger.info(f"✅ Generated synthetic engagement for {len(synthetic_df)} Khan Academy courses")
        
        return synthetic_df
    
    def _calculate_khan_completion_rate(self, engagement_score, difficulty, duration):
        """Calculate Khan Academy specific completion rate"""
        base_rate = 0.65
        
        # Engagement boost
        engagement_boost = (engagement_score - 0.5) * 0.25
        
        # Difficulty adjustment
        difficulty_adj = {'Beginner': 0.15, 'Intermediate': 0.0, 'Advanced': -0.10}.get(difficulty, 0.0)
        
        # Duration adjustment
        if duration <= 60:
            duration_adj = 0.1
        elif duration <= 120:
            duration_adj = 0.05
        else:
            duration_adj = -0.05
        
        final_rate = base_rate + engagement_boost + difficulty_adj + duration_adj
        return np.clip(final_rate, 0.3, 0.95)
    
    def _categorize_engagement(self, score):
        """Categorize engagement score"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def load_enhanced_data(self):
        """Load and enhance data with improved processing"""
        try:
            with self.app_context():
                # Load base data from database
                self.courses_df = pd.read_sql_query(
                    "SELECT * FROM courses_enhanced_analysis", 
                    self.db_engine
                )
                self.engagement_df = pd.read_sql_query(
                    "SELECT * FROM engagement_enhanced_analysis", 
                    self.db_engine
                )
                
                self.logger.info("=== ENHANCED DATA LOADING ===")
                self.logger.info(f"Courses loaded: {self.courses_df.shape}")
                self.logger.info(f"Engagement loaded: {self.engagement_df.shape}")
                
                # Fix platform column issues
                self._fix_platform_columns()
                
                # Generate Khan Academy synthetic data
                khan_engagement = self.generate_khan_academy_engagement(self.courses_df)
                
                # Combine engagement data
                if len(khan_engagement) > 0:
                    self.engagement_df = self._combine_engagement_data(self.engagement_df, khan_engagement)
                
                # Normalize engagement scores across platforms
                self.logger.info("🔄 Normalizing engagement scores across platforms...")
                self.engagement_df = self.normalize_engagement_scores(self.engagement_df)
                
                # Calculate enhanced completion rates
                self.logger.info("🔄 Calculating enhanced completion rates...")
                completion_rates = self.calculate_completion_rate_enhanced(self.engagement_df)
                self.engagement_df['completion_rate'] = completion_rates
                
                # Create merged datasets
                self.completion_df = pd.merge(
                    self.courses_df, self.engagement_df, 
                    on='course_id', how='inner', suffixes=('_course', '_engagement')
                )
                
                self.enhanced_df = pd.merge(
                    self.courses_df, self.engagement_df, 
                    on='course_id', how='left', suffixes=('_course', '_engagement')
                )
                
                # Fix platform columns after merge
                self._fix_merged_platform_columns()
                
                # Data quality summary
                self._print_data_quality_summary()
                
                return self.enhanced_df, self.completion_df
                
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced data loading: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _fix_platform_columns(self):
        """Fix platform column naming issues"""
        for df_name, df in [('courses_df', self.courses_df), ('engagement_df', self.engagement_df)]:
            if df is None:
                continue
                
            # Handle platform column variants
            if 'platform' not in df.columns:
                platform_candidates = [col for col in df.columns if 'platform' in col.lower()]
                if platform_candidates:
                    df['platform'] = df[platform_candidates[0]]
                    self.logger.info(f"✅ Fixed platform column in {df_name}")
            
            # Clean up duplicate columns
            platform_cols = [col for col in df.columns if col.startswith('platform') and col != 'platform']
            if platform_cols:
                df.drop(columns=platform_cols, inplace=True)
                self.logger.info(f"Dropped duplicate platform columns in {df_name}: {platform_cols}")
    
    def _combine_engagement_data(self, original_df, synthetic_df):
        """Combine original and synthetic engagement data"""
        # Ensure column compatibility
        all_columns = set(original_df.columns) | set(synthetic_df.columns)
        
        # Add missing columns to both dataframes
        for col in all_columns:
            if col not in original_df.columns:
                if col in ['is_synthetic']:
                    original_df[col] = False
                elif col in ['likes', 'comments', 'upvotes', 'gilded']:
                    original_df[col] = 0
                else:
                    original_df[col] = None
            
            if col not in synthetic_df.columns:
                if col in ['is_synthetic']:
                    synthetic_df[col] = True
                elif col in ['likes', 'comments', 'upvotes', 'gilded']:
                    synthetic_df[col] = 0
                else:
                    synthetic_df[col] = None
        
        # Combine datasets
        combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        self.logger.info(f"✅ Combined engagement data: {len(original_df)} + {len(synthetic_df)} = {len(combined_df)}")
        
        return combined_df
    
    def _fix_merged_platform_columns(self):
        """Fix platform columns after merging"""
        for df_name, df in [('completion_df', self.completion_df), ('enhanced_df', self.enhanced_df)]:
            if df is None:
                continue
                
            if 'platform_course' in df.columns and 'platform_engagement' in df.columns:
                df['platform'] = df['platform_course'].fillna(df['platform_engagement'])
                df.drop(columns=['platform_course', 'platform_engagement'], inplace=True)
                self.logger.info(f"✅ Fixed platform columns in {df_name}")
    
    def _print_data_quality_summary(self):
        """Print comprehensive data quality summary"""
        self.logger.info("\n=== DATA QUALITY SUMMARY ===")
        
        if self.engagement_df is not None:
            self.logger.info(f"Final engagement data shape: {self.engagement_df.shape}")
            
            # Platform distribution
            platform_counts = self.engagement_df['platform'].value_counts()
            self.logger.info(f"Platform distribution:\n{platform_counts}")
            
            # Completion rate statistics
            completion_stats = self.engagement_df['completion_rate'].describe()
            self.logger.info(f"\nCompletion rate statistics:\n{completion_stats}")
            
            # Engagement score statistics
            if 'engagement_score_normalized' in self.engagement_df.columns:
                engagement_stats = self.engagement_df['engagement_score_normalized'].describe()
                self.logger.info(f"\nNormalized engagement statistics:\n{engagement_stats}")
    
    def advanced_completion_analysis(self):
        """Advanced completion rate analysis"""
        try:
            if self.completion_df is None or len(self.completion_df) == 0:
                return {}
            
            analysis = {}
            
            # Overall completion statistics
            analysis['overall_stats'] = {
                'mean_completion_rate': self.completion_df['completion_rate'].mean(),
                'median_completion_rate': self.completion_df['completion_rate'].median(),
                'std_completion_rate': self.completion_df['completion_rate'].std(),
                'min_completion_rate': self.completion_df['completion_rate'].min(),
                'max_completion_rate': self.completion_df['completion_rate'].max()
            }
            
            # Platform-specific completion analysis
            platform_completion = self.completion_df.groupby('platform').agg({
                'completion_rate': ['mean', 'median', 'std', 'count'],
                'engagement_score_normalized': 'mean',
                'rating': 'mean'
            }).round(3)
            
            analysis['platform_completion'] = platform_completion.to_dict()
            
            # Difficulty level analysis
            if 'difficulty_level' in self.completion_df.columns:
                difficulty_completion = self.completion_df.groupby('difficulty_level').agg({
                    'completion_rate': ['mean', 'median', 'count'],
                    'engagement_score_normalized': 'mean'
                }).round(3)
                analysis['difficulty_completion'] = difficulty_completion.to_dict()
            
            # Duration impact analysis
            if 'duration_minutes' in self.completion_df.columns:
                # Create duration bins
                duration_bins = pd.cut(self.completion_df['duration_minutes'], 
                                     bins=[0, 30, 60, 120, 300, float('inf')], 
                                     labels=['0-30min', '30-60min', '60-120min', '120-300min', '300+min'])
                
                duration_analysis = self.completion_df.groupby(duration_bins).agg({
                    'completion_rate': ['mean', 'count'],
                    'engagement_score_normalized': 'mean'
                }).round(3)
                
                analysis['duration_impact'] = duration_analysis.to_dict()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in advanced completion analysis: {e}")
            return {}
    
    def generate_enhanced_report(self):
        """Generate comprehensive enhanced report"""
        try:
            # Load and process data
            enhanced_df, completion_df = self.load_enhanced_data()
            
            if enhanced_df is None or completion_df is None:
                return {}
            
            # Generate comprehensive analysis
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_courses': len(self.courses_df),
                    'total_engagement_records': len(self.engagement_df),
                    'platforms': sorted(self.courses_df['platform'].dropna().unique().tolist()),
                    'platform_counts': self.courses_df['platform'].value_counts().to_dict(),
                    'categories': self.courses_df['category'].nunique(),
                    'completion_data_available': len(completion_df)
                },
                'enhanced_completion_analysis': self.advanced_completion_analysis(),
                'engagement_patterns': self.engagement_pattern_mining(),
                'difficulty_calibration': self.content_difficulty_calibration(),
                'learning_paths': self.learning_path_optimization(),
                'behavioral_analytics': self.advanced_behavioral_analytics(),
                'data_quality_metrics': self._calculate_data_quality_metrics()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced report: {e}")
            return {}
    
    def _calculate_data_quality_metrics(self):
        """Calculate data quality metrics"""
        try:
            quality_metrics = {}
            
            if self.engagement_df is not None:
                # Completeness metrics
                total_records = len(self.engagement_df)
                completeness = {}
                
                key_columns = ['platform', 'engagement_score_normalized', 'completion_rate', 'rating']
                for col in key_columns:
                    if col in self.engagement_df.columns:
                        non_null_count = self.engagement_df[col].notna().sum()
                        completeness[col] = non_null_count / total_records
                
                quality_metrics['completeness'] = completeness
                
                # Consistency metrics
                consistency = {}
                
                # Check for reasonable ranges
                if 'completion_rate' in self.engagement_df.columns:
                    valid_completion = ((self.engagement_df['completion_rate'] >= 0) & 
                                       (self.engagement_df['completion_rate'] <= 1)).sum()
                    consistency['completion_rate_valid_range'] = valid_completion / total_records
                
                if 'rating' in self.engagement_df.columns:
                    valid_rating = ((self.engagement_df['rating'] >= 0) & 
                                   (self.engagement_df['rating'] <= 5)).sum()
                    consistency['rating_valid_range'] = valid_rating / total_records
                
                quality_metrics['consistency'] = consistency
                
                # Synthetic data ratio
                if 'is_synthetic' in self.engagement_df.columns:
                    synthetic_ratio = self.engagement_df['is_synthetic'].sum() / total_records
                    quality_metrics['synthetic_data_ratio'] = synthetic_ratio
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality metrics: {e}")
            return {}
    
    # Include all the original methods with improvementsf
    def learning_velocity_profiling(self):
        """Enhanced learning velocity profiling"""
        try:
            if self.completion_df is None or len(self.completion_df) == 0:
                return {}
            
            survival_data = self.completion_df.copy()
            survival_data['completed'] = (survival_data['completion_rate'] > 0.8).astype(int)
            survival_data['duration_days'] = np.random.exponential(45, len(survival_data))
            
            # Overall survival analysis
            kmf = KaplanMeierFitter()
            kmf.fit(survival_data['duration_days'], survival_data['completed'])
            
            # Platform-specific survival analysis
            platform_survival = {}
            for platform in survival_data['platform'].unique():
                if pd.notna(platform):
                    platform_data = survival_data[survival_data['platform'] == platform]
                    if len(platform_data) > 10:
                        platform_kmf = KaplanMeierFitter()
                        platform_kmf.fit(platform_data['duration_days'], platform_data['completed'])
                        
                        platform_survival[platform] = {
                            'median_survival_time': platform_kmf.median_survival_time_,
                            'survival_at_30_days': platform_kmf.survival_function_at_times(30).iloc[0],
                            'survival_at_60_days': platform_kmf.survival_function_at_times(60).iloc[0],
                            'total_courses': len(platform_data),
                            'average_completion_rate': platform_data['completion_rate'].mean()
                        }
            
            return {
                'overall_survival': {
                    'median_completion_time': kmf.median_survival_time_,
                    'completion_rate_30_days': kmf.survival_function_at_times(30).iloc[0] if kmf.median_survival_time_ else 0,
                    'completion_rate_60_days': kmf.survival_function_at_times(60).iloc[0] if kmf.median_survival_time_ else 0
                },
                'platform_comparison': platform_survival
            }
            
        except Exception as e:
            self.logger.error(f"Error in learning velocity profiling: {e}")
            return {}
        
    def _analyze_engagement_trends(self, engagement_data):
        try:
            platform_trends = {}
            for platform in engagement_data['platform'].unique():
                if pd.notna(platform):
                    platform_data = engagement_data[engagement_data['platform'] == platform]
                    if len(platform_data) > 5:
                        platform_trends[platform] = {
                            'avg_engagement': platform_data['engagement_score_normalized'].mean(),
                            'engagement_std': platform_data['engagement_score_normalized'].std(),
                            'high_engagement_ratio': (platform_data['engagement_score_normalized'] > 0.7).mean(),
                            'low_engagement_ratio': (platform_data['engagement_score_normalized'] < 0.3).mean()
                        }
            category_trends = {}
            if 'category' in engagement_data.columns:
                for category in engagement_data['category'].unique():
                    if pd.notna(category):
                        category_data = engagement_data[engagement_data['category'] == category]
                        if len(category_data) > 3:
                            category_trends[category] = {
                                'avg_engagement': category_data['engagement_score_normalized'].mean(),
                                'course_count': len(category_data),
                                'top_performer_ratio': (category_data['engagement_score_normalized'] > 0.8).mean()
                            }
            
            return {
                'platform_trends': platform_trends,
                'category_trends': category_trends
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing engagement trends: {e}")
            return {}
    
    def content_difficulty_calibration(self):
     try:
        desired_engagement_cols = ['course_id', 'engagement_score_normalized', 'completion_rate']
        available_engagement_cols = [col for col in desired_engagement_cols if col in self.engagement_df.columns]

        filtered_engagement_df = self.engagement_df[available_engagement_cols].copy()
        analysis_data = pd.merge(
            self.courses_df,
            filtered_engagement_df,
            on='course_id',
            how='inner'
        )

        if len(analysis_data) == 0:
            self.logger.warning("No merged data available for difficulty calibration")
            return {}

        difficulty_stats = {}
        if 'difficulty_level' in analysis_data.columns:
            for difficulty in analysis_data['difficulty_level'].dropna().unique():
                diff_data = analysis_data[analysis_data['difficulty_level'] == difficulty]

                difficulty_stats[difficulty] = {
                    'course_count': len(diff_data),
                    'avg_duration': diff_data['duration_minutes'].mean(),
                    'avg_engagement': diff_data['engagement_score_normalized'].mean() if 'engagement_score_normalized' in diff_data.columns else None,
                    'completion_rate': diff_data['completion_rate'].mean() if 'completion_rate' in diff_data.columns else None,
                    'avg_price': diff_data['price'].mean() if 'price' in diff_data.columns else None,
                    'free_course_ratio': diff_data['is_free'].mean() if 'is_free' in diff_data.columns else None
                }

        statistical_tests = {}
        if len(difficulty_stats) >= 2:
            difficulties = list(difficulty_stats.keys())
            duration_groups = [
                analysis_data[analysis_data['difficulty_level'] == diff]['duration_minutes'].dropna()
                for diff in difficulties
            ]
            duration_groups = [group for group in duration_groups if len(group) > 3]

            if len(duration_groups) >= 2:
                try:
                    h_stat, p_value = stats.kruskal(*duration_groups)
                    statistical_tests['duration_difference'] = {
                        'test': 'Kruskal-Wallis',
                        'statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    self.logger.warning(f"Statistical test failed: {e}")

        complexity_scores = self._calculate_content_complexity(analysis_data)

        # ✅ Platform Stats Section
        platform_stats = {}
        if 'platform' in analysis_data.columns:
            for platform in analysis_data['platform'].dropna().unique():
                platform_data = analysis_data[analysis_data['platform'] == platform]
                platform_stats[platform] = {
                    'course_count': len(platform_data),
                    'avg_completion_rate': platform_data['completion_rate'].mean() if 'completion_rate' in platform_data.columns else None,
                    'avg_engagement': platform_data['engagement_score_normalized'].mean() if 'engagement_score_normalized' in platform_data.columns else None,
                    'avg_duration': platform_data['duration_minutes'].mean() if 'duration_minutes' in platform_data.columns else None,
                    'avg_price': platform_data['price'].mean() if 'price' in platform_data.columns else None,
                    'free_course_ratio': platform_data['is_free'].mean() if 'is_free' in platform_data.columns else None
                }

        return {
            'difficulty_statistics': difficulty_stats,
            'statistical_tests': statistical_tests,
            'complexity_scores': complexity_scores,
            'platform_completion_stats': platform_stats  # ✅ Added return key
        }

     except Exception as e:
        self.logger.error(f"Error in content difficulty calibration: {e}")
        return {}

    def _calculate_content_complexity(self, data):
        try:
            complexity_factors = {}
            if 'duration_minutes' in data.columns:
                duration_data = data['duration_minutes'].dropna()
                complexity_factors['duration'] = {
                    'mean': duration_data.mean(),
                    'std': duration_data.std(),
                    'complexity_score': (duration_data > duration_data.quantile(0.75)).mean()
                }
            if 'description' in data.columns:
                desc_lengths = data['description'].str.len().fillna(0)
                complexity_factors['description_complexity'] = {
                    'avg_length': desc_lengths.mean(),
                    'complexity_score': (desc_lengths > desc_lengths.quantile(0.75)).mean()
                }
            if 'engagement_score_normalized' in data.columns and 'duration_minutes' in data.columns:
                correlation = data['engagement_score_normalized'].corr(data['duration_minutes'])
                complexity_factors['engagement_duration_correlation'] = {
                    'correlation': correlation,
                    'interpretation': 'positive' if correlation > 0.1 else 'negative' if correlation < -0.1 else 'neutral'
                }
            return complexity_factors
        except Exception as e:
            self.logger.error(f"Error calculating content complexity: {e}")
            return {}
    
    def learning_path_optimization(self):
        try:
            G = nx.DiGraph()
            for _, course in self.courses_df.iterrows():
                G.add_node(course['course_id'], 
                          title=course['title'],
                          category=course['category'],
                          difficulty=course['difficulty_level'],
                          duration=course['duration_minutes'],
                          platform=course['platform'])
            
            courses_by_category = self.courses_df.groupby('category')
            for category, category_courses in courses_by_category:
                if len(category_courses) > 1:
                    difficulty_order = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
                    category_courses['difficulty_num'] = category_courses['difficulty_level'].map(difficulty_order).fillna(1)
                    category_courses = category_courses.sort_values(['difficulty_num', 'duration_minutes'])
                    
                    for i in range(len(category_courses) - 1):
                        current_course = category_courses.iloc[i]
                        next_course = category_courses.iloc[i + 1]
                        
                        if (current_course['difficulty_num'] <= next_course['difficulty_num']):
                            G.add_edge(current_course['course_id'], next_course['course_id'],
                                     weight=1.0, relationship='prerequisite')
            network_metrics = self._calculate_network_metrics(G)
            optimal_paths = self._find_optimal_paths(G)
            return {
                'network_metrics': network_metrics,
                'optimal_paths': optimal_paths,
                'network_size': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
            }
        except Exception as e:
            self.logger.error(f"Error in learning path optimization: {e}")
            return {}
    
    def _calculate_network_metrics(self, G):
        try:
            metrics = {}
            metrics['density'] = nx.density(G)
            metrics['is_connected'] = nx.is_weakly_connected(G)
            if G.number_of_nodes() > 0:
                in_degree_centrality = nx.in_degree_centrality(G)
                out_degree_centrality = nx.out_degree_centrality(G)
                metrics['most_prerequisite_courses'] = sorted(
                    in_degree_centrality.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                metrics['most_foundational_courses'] = sorted(
                    out_degree_centrality.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            if G.number_of_nodes() > 0:
                components = list(nx.weakly_connected_components(G))
                metrics['connected_components'] = len(components)
                metrics['largest_component_size'] = max(len(comp) for comp in components) if components else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating network metrics: {e}")
            return {}
    
    def _find_optimal_paths(self, G):
        try:
            optimal_paths = {}
            categories = set(nx.get_node_attributes(G, 'category').values())
            
            for category in categories:
                if pd.notna(category):
                    category_nodes = [node for node, attr in G.nodes(data=True) 
                                    if attr.get('category') == category]
                    
                    if len(category_nodes) > 1:
                        beginners = [node for node in category_nodes if G.in_degree(node) == 0]
                        advanced = [node for node in category_nodes if G.out_degree(node) == 0]
                        paths = []
                        for begin in beginners:
                            for end in advanced:
                                try:
                                    if nx.has_path(G, begin, end):
                                        path = nx.shortest_path(G, begin, end)
                                        paths.append(path)
                                except:
                                    continue
                        if paths:
                            optimal_paths[category] = {
                                'paths': paths[:3],  # Top 3 paths
                                'beginner_courses': beginners,
                                'advanced_courses': advanced,
                                'total_courses': len(category_nodes)
                            }
            
            return optimal_paths
        except Exception as e:
            self.logger.error(f"Error finding optimal paths: {e}")
            return {}
    
    def advanced_behavioral_analytics(self):
        try:
            attention_analysis = self._model_attention_span()
            retention_analysis = self._model_retention()
            return {
                'attention_span_analysis': attention_analysis,
                'retention_analysis': retention_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced behavioral analytics: {e}")
            return {}
    
    def _model_attention_span(self):
        try:
            analysis_data = pd.merge(
                self.courses_df[['course_id', 'duration_minutes']], 
                self.engagement_df[['course_id', 'engagement_score_normalized']], 
                on='course_id', 
                how='inner'
            )
            
            if len(analysis_data) == 0:
                return {}
            durations = analysis_data['duration_minutes'].values
            engagements = analysis_data['engagement_score_normalized'].values
            valid_mask = (durations > 0) & (engagements > 0)
            if valid_mask.sum() < 10:
                return {}
            
            valid_durations = durations[valid_mask]
            valid_engagements = engagements[valid_mask]
            try:
                log_engagements = np.log(valid_engagements + 1e-10)  # Avoid log(0)
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_durations, log_engagements)
                
                return {
                    'decay_rate': -slope,
                    'initial_engagement': np.exp(intercept),
                    'correlation': r_value,
                    'p_value': p_value,
                    'optimal_duration': -1 / slope if slope < 0 else None,
                    'model_quality': r_value**2
                }
            except:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error modeling attention span: {e}")
            return {}
    
    def _model_retention(self):
        try:
            retention_data = self.engagement_df.copy()
            if 'engagement_score_normalized' not in retention_data.columns:
                return {}
            engagements = retention_data['engagement_score_normalized'].values
            time_points = np.array([1, 7, 30, 90, 365])  # 1 day, 1 week, 1 month, 3 months, 1 year
            retention_curves = {}
            high_engagement_threshold = np.percentile(engagements, 75)
            high_engagement_courses = engagements >= high_engagement_threshold
            medium_engagement_courses = (engagements >= np.percentile(engagements, 25)) & (engagements < high_engagement_threshold)
            low_engagement_courses = engagements < np.percentile(engagements, 25)
            decay_rates = {
                'high_engagement': 0.05,    # Slow decay
                'medium_engagement': 0.1,   # Medium decay
                'low_engagement': 0.2       # Fast decay
            }
            for level, decay_rate in decay_rates.items():
                initial_strength = np.mean(engagements[eval(f"{level}_courses")])
                retention_curve = initial_strength * np.exp(-decay_rate * time_points)
                
                retention_curves[level] = {
                    'initial_strength': initial_strength,
                    'decay_rate': decay_rate,
                    'retention_at_1_day': retention_curve[0],
                    'retention_at_1_week': retention_curve[1],
                    'retention_at_1_month': retention_curve[2],
                    'retention_at_3_months': retention_curve[3],
                    'retention_at_1_year': retention_curve[4],
                    'half_life_days': np.log(2) / decay_rate
                }
            return retention_curves
            
        except Exception as e:
            self.logger.error(f"Error modeling retention: {e}")
            return {}
    
    def generate_comprehensive_report(self):
        try:
            self.load_enhanced_data()
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_courses': len(self.courses_df),
                    'total_engagement_records': len(self.engagement_df),
                    'platforms': self.courses_df['platform'].nunique(),
                    'categories': self.courses_df['category'].nunique()
                },
                'learning_velocity': self.learning_velocity_profiling(),
                'difficulty_calibration': self.content_difficulty_calibration(),
                'learning_paths': self.learning_path_optimization(),
                'behavioral_analytics': self.advanced_behavioral_analytics()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {}
    
    def save_analysis_results(self, results):
        try:
                results_df = pd.DataFrame([{
                    'analysis_date': datetime.now(),
                    'analysis_type': 'learning_intelligence',
                    'results': str(results),  # Store as string for simplicity
                    'total_courses_analyzed': len(self.courses_df),
                    'total_engagement_records': len(self.engagement_df)
                }])
                
                results_df.to_sql(
                    'learning_intelligence_results',
                    self.db_engine,
                    if_exists='append',
                    index=False
                )
                
                self.logger.info("Learning intelligence results saved to database")
                
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {e}")

def run_learning_intelligence_analysis(db_uri=None):
    engine = LearningIntelligenceEngine(db_uri)
    
    print("\n" + "="*60)
    print("LEARNING INTELLIGENCE ANALYSIS")
    print("="*60)
    
    report = engine.generate_comprehensive_report()

    # === Inject synthetic completion_times ===
    if 'learning_velocity' in report and 'overall_survival' in report['learning_velocity']:
        try:
            df = engine.enhanced_df.copy()
            df['completion_time_est'] = df.apply(
                lambda row: row['duration_minutes'] / (row['completion_rate'] / 100) / 60
                if row['completion_rate'] > 0 else None,
                axis=1
            )
            completion_times = df['completion_time_est'].dropna().tolist()

            report['learning_velocity']['overall_survival']['completion_times'] = completion_times
            print(f"✅ Injected {len(completion_times)} estimated completion times into report")
        except Exception as e:
            print(f"⚠️ Failed to calculate completion_times: {e}")

    engine.save_analysis_results(report)
    
    if report:
        print(f"Analysis completed for {report['data_summary']['total_courses']} courses")
        print(f"Platforms analyzed: {report['data_summary']['platforms']}")
        print(f"Categories analyzed: {report['data_summary']['categories']}")
        
        if 'learning_velocity' in report:
            velocity = report['learning_velocity']
            if 'overall_survival' in velocity:
                print(f"Median completion time: {velocity['overall_survival'].get('median_completion_time', 'N/A')} days")
        
        if 'learning_paths' in report:
            paths = report['learning_paths']
            if 'network_size' in paths:
                print(f"Learning network: {paths['network_size']['nodes']} courses, {paths['network_size']['edges']} connections")
        
        if 'behavioral_analytics' in report:
            behavioral = report['behavioral_analytics']
            if 'attention_span_analysis' in behavioral:
                attention = behavioral['attention_span_analysis']
                if 'decay_rate' in attention:
                    print(f"Attention decay rate: {attention['decay_rate']:.4f}")
                if 'optimal_duration' in attention and attention['optimal_duration']:
                    print(f"Optimal course duration: {attention['optimal_duration']:.1f} minutes")
    
    return report


def create_learning_dashboard_data(report):
    dashboard_data = {
        'summary_metrics': {
            'total_courses': report.get('data_summary', {}).get('total_courses', 0),
            'total_platforms': report.get('data_summary', {}).get('platforms', 0),
            'total_categories': report.get('data_summary', {}).get('categories', 0),
            'analysis_timestamp': report.get('timestamp', datetime.now().isoformat())
        },
        'engagement_insights': {},
        'learning_velocity_insights': {},
        'difficulty_insights': {},
        'path_optimization': {},
        'behavioral_insights': {}
    }
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        dashboard_data['learning_velocity_insights'] = {
            'overall_survival': velocity.get('overall_survival', {}),
            'platform_comparison': velocity.get('platform_comparison', {}),
            'cox_regression': velocity.get('cox_regression', {})
        }
    if 'difficulty_calibration' in report:
        difficulty = report['difficulty_calibration']
        dashboard_data['difficulty_insights'] = {
            'difficulty_statistics': difficulty.get('difficulty_statistics', {}),
            'statistical_tests': difficulty.get('statistical_tests', {}),
            'complexity_scores': difficulty.get('complexity_scores', {}),
            'platform_completion_stats': difficulty.get('platform_completion_stats', {})  # ✅ ADD THIS

        }
    
    if 'learning_paths' in report:
        paths = report['learning_paths']
        dashboard_data['path_optimization'] = {
            'network_metrics': paths.get('network_metrics', {}),
            'optimal_paths': paths.get('optimal_paths', {}),
            'network_size': paths.get('network_size', {})
        }
    
    if 'behavioral_analytics' in report:
        behavioral = report['behavioral_analytics']
        dashboard_data['behavioral_insights'] = {
            'attention_span': behavioral.get('attention_span_analysis', {}),
            'success_patterns': behavioral.get('success_patterns', {}),
            'retention_analysis': behavioral.get('retention_analysis', {})
        }
    
    return dashboard_data
def generate_learning_recommendations(report):
    recommendations = {
        'course_optimization': [],
        'engagement_improvement': [],
        'learning_path_enhancement': [],
        'platform_strategy': [],
        'content_strategy': []
    }
    if 'difficulty_calibration' in report:
        difficulty = report['difficulty_calibration']
        if 'difficulty_statistics' in difficulty:
            for diff_level, stats in difficulty['difficulty_statistics'].items():
                if stats['avg_engagement'] is not None and stats['avg_engagement'] < 0.5:
                    recommendations['course_optimization'].append(
                        f"Review {diff_level} level courses - low engagement ({stats['avg_engagement']:.2f})"
                    )
                if stats['avg_duration'] > 180:  # 3+ hours
                    recommendations['course_optimization'].append(
                        f"Consider breaking down {diff_level} courses - average duration {stats['avg_duration']:.0f} minutes"
                    )
    
    if 'learning_paths' in report:
        paths = report['learning_paths']
        if 'network_metrics' in paths:
            metrics = paths['network_metrics']
            if metrics.get('density', 0) < 0.1:
                recommendations['learning_path_enhancement'].append(
                    "Network density is low - consider creating more prerequisite relationships"
                )
            if metrics.get('connected_components', 0) > 5:
                recommendations['learning_path_enhancement'].append(
                    "Multiple disconnected learning paths - consider bridging courses"
                )
    
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        if 'platform_comparison' in velocity:
            platform_performances = []
            for platform, data in velocity['platform_comparison'].items():
                survival_30 = data.get('survival_at_30_days', 0)
                platform_performances.append((platform, survival_30))
            
            if platform_performances:
                platform_performances.sort(key=lambda x: x[1], reverse=True)
                best_platform = platform_performances[0][0]
                worst_platform = platform_performances[-1][0]
                
                recommendations['platform_strategy'].append(
                    f"Focus on {best_platform} platform - highest completion rate"
                )
                if len(platform_performances) > 1:
                    recommendations['platform_strategy'].append(
                        f"Investigate {worst_platform} platform issues - lowest completion rate"
                    )
    
    if 'behavioral_analytics' in report:
        behavioral = report['behavioral_analytics']
        if 'attention_span_analysis' in behavioral:
            attention = behavioral['attention_span_analysis']
            if 'optimal_duration' in attention and attention['optimal_duration']:
                optimal_duration = attention['optimal_duration']
                recommendations['content_strategy'].append(
                    f"Optimize course duration around {optimal_duration:.0f} minutes for maximum engagement"
                )
            if 'decay_rate' in attention and attention['decay_rate'] > 0.15:
                recommendations['content_strategy'].append(
                    "High attention decay rate - consider more interactive content"
                )
        
        if 'retention_analysis' in behavioral:
            retention = behavioral['retention_analysis']
            for level, data in retention.items():
                if 'retention_at_1_month' in data and data['retention_at_1_month'] < 0.3:
                    recommendations['content_strategy'].append(
                        f"Improve {level.replace('_', ' ')} retention - only {data['retention_at_1_month']:.1%} retained after 1 month"
                    )
    return recommendations

def export_analysis_summary(report, filename=None):
    if filename is None:
        filename = f"learning_intelligence_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary = {
        'Analysis Overview': {
            'Total Courses Analyzed': report.get('data_summary', {}).get('total_courses', 0),
            'Total Platforms': report.get('data_summary', {}).get('platforms', 0),
            'Total Categories': report.get('data_summary', {}).get('categories', 0),
            'Analysis Date': report.get('timestamp', datetime.now().isoformat())
        }
    }
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        if 'overall_survival' in velocity:
            summary['Learning Velocity'] = {
                'Median Completion Time (days)': velocity['overall_survival'].get('median_completion_time', 'N/A'),
                'Completion Rate at 30 Days': velocity['overall_survival'].get('completion_rate_30_days', 'N/A'),
                'Completion Rate at 60 Days': velocity['overall_survival'].get('completion_rate_60_days', 'N/A')
            }
    
    if 'difficulty_calibration' in report:
        difficulty = report['difficulty_calibration']
        summary['Difficulty Analysis'] = {
            'Difficulty Levels Analyzed': len(difficulty.get('difficulty_statistics', {})),
            'Statistical Tests Performed': len(difficulty.get('statistical_tests', {}))
        }
    if 'learning_paths' in report:
        paths = report['learning_paths']
        network_size = paths.get('network_size', {})
        summary['Learning Paths'] = {
            'Network Nodes': network_size.get('nodes', 0),
            'Network Edges': network_size.get('edges', 0),
            'Optimal Paths Found': len(paths.get('optimal_paths', {}))
        }
    if 'behavioral_analytics' in report:
        behavioral = report['behavioral_analytics']
        summary['Behavioral Analytics'] = {
            'Attention Span Modeled': bool(behavioral.get('attention_span_analysis', {})),
            'Success Patterns Identified': bool(behavioral.get('success_patterns', {})),
            'Retention Analysis Complete': bool(behavioral.get('retention_analysis', {}))
        }
    recommendations = generate_learning_recommendations(report)
    summary['Recommendations'] = recommendations
    return summary
def get_learning_velocity():
    velocity = dashboard_data.get('learning_velocity_insights', {})
    survival = velocity.get('overall_survival', {})
    
    print(f"DEBUG: velocity keys: {velocity.keys()}")
    print(f"DEBUG: survival keys: {survival.keys()}")
    print(f"DEBUG: completion_times length: {len(survival.get('completion_times', []))}")
    
    return {
        "median_completion_time": survival.get('median_completion_time', 'N/A'),
        "completion_rate_30_days": survival.get('completion_rate_30_days', 'N/A'),
        "completion_rate_60_days": survival.get('completion_rate_60_days', 'N/A'),
        "cox_regression": velocity.get('cox_regression', {}),
        "platform_comparison": velocity.get('platform_comparison', {}),
        "completion_times": survival.get('completion_times', []),  # This might be empty
        "completed": survival.get('completed', [])
    }

report = run_learning_intelligence_analysis()
dashboard_data = create_learning_dashboard_data(report)
recommendations = generate_learning_recommendations(report)

if __name__ == "__main__":
   
    checking=get_learning_velocity()
    print(f"DEBUG: {checking}")