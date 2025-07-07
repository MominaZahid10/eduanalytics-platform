import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
from sqlalchemy import text
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import logging
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

class AdvancedEduDataQuality:
    def __init__(self, app=None):
        self.app = app
        self.db = None
        if app:
            self.init_app(app)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            'completeness_threshold': 0.7,
            'outlier_threshold': 0.05,
            'consistency_threshold': 0.95,
            'min_sample_size': 200
        }

    def init_app(self, app):
        from extensions import db
        self.app = app
        self.db = db

    def comprehensive_quality_assessment(self):
        self.logger.info("Starting comprehensive data quality assessment...")
        self._load_enhanced_data()
        quality_report = {
            'completeness': self._assess_completeness(),
            'consistency': self._assess_consistency(),
            'validity': self._assess_validity(),
            'statistical_power': self._assess_statistical_power(),
            'recommendations': []
        }
        quality_report['recommendations'] = self._generate_recommendations(quality_report)
        
        return quality_report

    def _load_enhanced_data(self):
        try:
            with self.app.app_context():
                courses_query = text("""
                    SELECT 
                        c.id as course_id,
                        c.title,
                        c.instructor,
                        c.description,
                        c.url,
                        COALESCE(c.duration_minutes, 0) as duration_minutes,
                        COALESCE(c.difficulty_level, 'Unknown') as difficulty_level,
                        c.category,
                        COALESCE(c.language, 'Unknown') as language,
                        COALESCE(c.price, 0.0) as price,
                        COALESCE(c.currency, 'USD') as currency,
                        c.is_free,
                        c.published_at,
                        c.created_at,
                        c.tags,
                        p.name as platform,
                        p.base_url as platform_url
                    FROM courses c
                    JOIN platforms p ON c.platform_id = p.id
                    WHERE p.is_active = true
                """)

                self.courses_df = pd.read_sql(courses_query, self.db.engine)
                engagement_query = text("""
                    SELECT 
                        em.course_id,
                        em.metric_type,
                        COALESCE(em.value, 0) as value,
                        em.recorded_at,
                        c.title as course_title
                    FROM engagement_metrics em
                    JOIN courses c ON em.course_id = c.id
                    WHERE em.value IS NOT NULL
                    ORDER BY em.course_id, em.recorded_at
                """)

                self.engagement_df = pd.read_sql(engagement_query, self.db.engine)
                self.engagement_pivot = self.engagement_df.pivot_table(
                    index=['course_id', 'course_title'],
                    columns='metric_type',
                    values='value',
                    aggfunc='last',
                    fill_value=0
                ).reset_index()

                self.logger.info(f"Loaded {len(self.courses_df)} courses and {len(self.engagement_df)} engagement records")
        except Exception as e:
            self.logger.error(f"Error loading enhanced data: {e}")
            raise

    def _assess_completeness(self):
        completeness_scores = {}
        critical_fields = {
            'title': 1.0,         
            'instructor': 0.8,     
            'duration_minutes': 0.7, 
            'difficulty_level': 0.6, 
            'description': 0.5,   
        }
        for field, min_threshold in critical_fields.items():
            if field in self.courses_df.columns:
                completeness = 1 - (self.courses_df[field].isnull().sum() / len(self.courses_df))
                completeness_scores[field] = {
                    'score': completeness,
                    'threshold': min_threshold,
                    'status': 'PASS' if completeness >= min_threshold else 'FAIL',
                    'missing_count': self.courses_df[field].isnull().sum()
                }
        engagement_coverage = len(self.engagement_pivot) / len(self.courses_df)
        completeness_scores['engagement_coverage'] = {
            'score': engagement_coverage,
            'threshold': 0.6,
            'status': 'PASS' if engagement_coverage >= 0.6 else 'FAIL',
            'courses_with_data': len(self.engagement_pivot)
        }
        return completeness_scores

    def _assess_consistency(self):
        consistency_issues = []
        free_but_priced = self.courses_df[
            (self.courses_df['is_free'] == True) & 
            (self.courses_df['price'] > 0)
        ]
        if len(free_but_priced) > 0:
            consistency_issues.append({
                'type': 'price_consistency',
                'count': len(free_but_priced),
                'description': 'Courses marked as free but have price > 0'
            })
        duration_outliers = self.courses_df[
            (self.courses_df['duration_minutes'] > 0) & 
            ((self.courses_df['duration_minutes'] < 1) | 
             (self.courses_df['duration_minutes'] > 2000))
        ]
        if len(duration_outliers) > 0:
            consistency_issues.append({
                'type': 'duration_outliers',
                'count': len(duration_outliers),
                'description': 'Courses with unrealistic durations (<1 min or >33 hours)'
            })
        platform_consistency = self._check_platform_consistency()
        return {
            'issues': consistency_issues,
            'platform_consistency': platform_consistency,
            'overall_score': max(0, 1 - (len(consistency_issues) / len(self.courses_df)))
        }

    def _check_platform_consistency(self):
        platform_stats = {}
        for platform in self.courses_df['platform'].unique():
            platform_data = self.courses_df[self.courses_df['platform'] == platform]
            platform_stats[platform] = {
                'course_count': len(platform_data),
                'avg_duration': platform_data['duration_minutes'].mean(),
                'price_consistency': (platform_data['is_free'] == (platform_data['price'] == 0)).mean(),
                'completeness': 1 - (platform_data.isnull().sum().sum() / platform_data.size),
                'engagement_coverage': len(
                    self.engagement_pivot[self.engagement_pivot['course_id'].isin(platform_data['course_id'])]
                ) / len(platform_data) if len(platform_data) > 0 else 0
            }
        return platform_stats

    def _assess_validity(self):
        validity_issues = []
        valid_duration = self.courses_df[
            (self.courses_df['duration_minutes'] >= 1) & 
            (self.courses_df['duration_minutes'] <= 1440)
        ]
        duration_validity = len(valid_duration) / len(self.courses_df[self.courses_df['duration_minutes'] > 0])
        valid_price = self.courses_df[
            (self.courses_df['price'] >= 0) & 
            (self.courses_df['price'] <= 10000)
        ]
        price_validity = len(valid_price) / len(self.courses_df[self.courses_df['price'].notna()])
        engagement_validity = self._assess_engagement_validity()
        return {
            'duration_validity': duration_validity,
            'price_validity': price_validity,
            'engagement_validity': engagement_validity,
            'overall_validity': np.mean([duration_validity, price_validity, engagement_validity])
        }

    def _assess_engagement_validity(self):
        if len(self.engagement_pivot) == 0:
            return 0.0
        valid_scores = []
        positive_metrics = ['views', 'likes', 'comments', 'enrollments']
        for metric in positive_metrics:
            if metric in self.engagement_pivot.columns:
                valid_count = (self.engagement_pivot[metric] >= 0).sum()
                total_count = self.engagement_pivot[metric].notna().sum()
                if total_count > 0:
                    valid_scores.append(valid_count / total_count)
        rate_metrics = ['upvote_ratio', 'completion_rate']
        for metric in rate_metrics:
            if metric in self.engagement_pivot.columns:
                valid_count = (
                    (self.engagement_pivot[metric] >= 0) & 
                    (self.engagement_pivot[metric] <= 1)
                ).sum()
                total_count = self.engagement_pivot[metric].notna().sum()
                if total_count > 0:
                    valid_scores.append(valid_count / total_count)
        return np.mean(valid_scores) if valid_scores else 1.0

    def _assess_statistical_power(self):
        power_assessment = {}
        total_courses = len(self.courses_df)
        power_assessment['total_sample'] = {
            'size': total_courses,
            'adequate': total_courses >= self.quality_thresholds['min_sample_size'],
            'recommended_minimum': self.quality_thresholds['min_sample_size']
        }
        platform_sizes = self.courses_df['platform'].value_counts()
        power_assessment['platform_samples'] = {}
        for platform, size in platform_sizes.items():
            adequate = size >= 30 
            power_assessment['platform_samples'][platform] = {
                'size': size,
                'adequate': adequate,
                'recommended_minimum': 30
            }
        engagement_courses = len(self.engagement_pivot)
        power_assessment['engagement_sample'] = {
            'size': engagement_courses,
            'adequate': engagement_courses >= 50,
            'coverage': engagement_courses / total_courses if total_courses > 0 else 0
        }
        return power_assessment

    def _generate_recommendations(self, quality_report):
        recommendations = []
        completeness = quality_report['completeness']
        for field, metrics in completeness.items():
            if metrics['status'] == 'FAIL':
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Data Collection',
                    'issue': f'Low completeness for {field}',
                    'recommendation': f'Improve data collection for {field} - currently {metrics["score"]:.1%}, need {metrics["threshold"]:.1%}',
                    'impact': 'Limits analytical capabilities and statistical power'
                })
        power = quality_report['statistical_power']
        if not power['total_sample']['adequate']:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Sample Size',
                'issue': 'Insufficient sample size for robust analysis',
                'recommendation': f'Collect data for at least {power["total_sample"]["recommended_minimum"]} courses',
                'impact': 'Current sample size may lead to unreliable statistical conclusions'
            })
        platform_samples = power['platform_samples']
        unbalanced_platforms = [p for p, metrics in platform_samples.items() if not metrics['adequate']]
        if unbalanced_platforms:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Data Balance',
                'issue': f'Insufficient data for platforms: {", ".join(unbalanced_platforms)}',
                'recommendation': 'Collect more courses from underrepresented platforms for balanced analysis',
                'impact': 'Platform comparisons may be statistically unreliable'
            })
        consistency = quality_report['consistency']
        if consistency['overall_score'] < 0.95:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Quality',
                'issue': 'Data consistency issues detected',
                'recommendation': 'Implement data validation rules and fix inconsistencies',
                'impact': 'Inconsistent data can lead to incorrect insights'
            })
        return recommendations
    def enhance_descriptions(self):
        enhanced_df = self.courses_df.copy()
        youtube_mask = (enhanced_df['platform'] == 'YouTube') & (enhanced_df['description'].isna())
        enhanced_df.loc[youtube_mask, 'description'] = enhanced_df.loc[youtube_mask, 'title'].apply(
            self._generate_youtube_description
        )
        title_mask = enhanced_df['description'].isna() & (enhanced_df['title'].str.len() > 20)
        enhanced_df.loc[title_mask, 'description'] = enhanced_df.loc[title_mask].apply(
            self._generate_course_description, axis=1
        )
        category_mask = enhanced_df['description'].isna() & enhanced_df['category'].notna()
        enhanced_df.loc[category_mask, 'description'] = enhanced_df.loc[category_mask].apply(
            self._generate_category_description, axis=1
        )
        instructor_mask = enhanced_df['description'].isna() & enhanced_df['instructor'].notna()
        enhanced_df.loc[instructor_mask, 'description'] = enhanced_df.loc[instructor_mask].apply(
            self._generate_instructor_description, axis=1
        )
        remaining_mask = enhanced_df['description'].isna()
        enhanced_df.loc[remaining_mask, 'description'] = enhanced_df.loc[remaining_mask, 'title'].apply(
            lambda x: f"Educational content covering {str(x).lower()}. Learn key concepts and practical applications."
        )
        self.enhanced_courses_df = enhanced_df
        original_completeness = 1 - (self.courses_df['description'].isnull().sum() / len(self.courses_df))
        new_completeness = 1 - (enhanced_df['description'].isnull().sum() / len(enhanced_df))
        self.logger.info(f"Description completeness improved from {original_completeness:.1%} to {new_completeness:.1%}")
        return enhanced_df

    def _generate_youtube_description(self, title):
        title_lower = str(title).lower()
        if any(word in title_lower for word in ['tutorial', 'how to', 'guide', 'learn']):
            return f"Step-by-step tutorial on {title}. Perfect for beginners and intermediate learners."
        elif any(word in title_lower for word in ['review', 'analysis', 'explained']):
            return f"Comprehensive review and analysis of {title}. Key concepts explained clearly."
        elif any(word in title_lower for word in ['intro', 'introduction', 'basics']):
            return f"Introduction to {title}. Master the fundamentals with clear explanations."
        elif any(word in title_lower for word in ['advanced', 'master', 'expert']):
            return f"Advanced course on {title}. Develop expert-level skills and knowledge."
        else:
            return f"Educational video covering {title}. Learn practical skills and key concepts."

    def _generate_course_description(self, row):
        title = str(row['title'])
        category = str(row['category']) if pd.notna(row['category']) else 'general topics'
        difficulty = str(row['difficulty_level']) if pd.notna(row['difficulty_level']) else 'all levels'
        duration = row['duration_minutes'] if pd.notna(row['duration_minutes']) and row['duration_minutes'] > 0 else None
        base_desc = f"Comprehensive course on {title} in {category}. "
        base_desc += f"Designed for {difficulty} learners. "
        if duration:
            if duration < 60:
                base_desc += "Quick and focused content. "
            elif duration > 300:
                base_desc += "In-depth coverage with extensive material. "
            else:
                base_desc += "Well-structured content with practical examples. "
        base_desc += "Build practical skills and theoretical understanding."
        return base_desc

    def _generate_category_description(self, row):
        category = str(row['category'])
        title = str(row['title'])
        category_templates = {
            'Programming': f"Programming course covering {title}. Learn coding fundamentals, best practices, and hands-on development skills.",
            'Data Science': f"Data science course on {title}. Master data analysis, visualization, and statistical methods.",
            'Mathematics': f"Mathematics course covering {title}. Develop problem-solving skills and mathematical reasoning.",
            'Science': f"Science course on {title}. Explore scientific principles, theories, and real-world applications.",
            'Business': f"Business course covering {title}. Learn practical business skills, strategies, and management techniques.",
            'Arts': f"Arts course on {title}. Develop creative skills, artistic techniques, and cultural understanding.",
            'Language': f"Language course covering {title}. Improve communication skills, grammar, and cultural fluency.",
            'Engineering': f"Engineering course on {title}. Master technical concepts, design principles, and practical applications.",
            'Medicine': f"Medical course covering {title}. Learn clinical knowledge, diagnostic skills, and healthcare practices.",
            'Technology': f"Technology course on {title}. Understand technical concepts, tools, and industry applications."
        }
        return category_templates.get(category, f"Educational course on {title} in {category}. Comprehensive learning experience.")

    def _generate_instructor_description(self, row):
        instructor = str(row['instructor'])
        title = str(row['title'])
        category = str(row['category']) if pd.notna(row['category']) else 'this subject'
        return f"Expert-led course on {title} by {instructor}. Learn from industry professionals with real-world experience in {category}."

    def smart_data_imputation(self):
        imputed_df = self.courses_df.copy()
        imputed_df = self._impute_duration_smart(imputed_df)
        imputed_df['difficulty_level'] = self._impute_difficulty_level_smart(imputed_df)
        imputed_df = self._impute_language_smart(imputed_df)
        imputed_df = self._impute_category_smart(imputed_df)
        imputed_df = self._impute_instructor_smart(imputed_df)
        imputed_df = self._fix_price_consistency(imputed_df)
        self.courses_imputed = imputed_df
        self._log_imputation_improvements(imputed_df)
        return imputed_df

    def _impute_duration_smart(self, df):
        df_work = df.copy()
        df_work['title_length'] = df_work['title'].str.len()
        df_work['has_description'] = df_work['description'].notna().astype(int)
        df_work['desc_length'] = df_work['description'].str.len().fillna(0)
        le_platform = LabelEncoder()
        le_category = LabelEncoder()
        le_difficulty = LabelEncoder()
        df_work['platform_encoded'] = le_platform.fit_transform(df_work['platform'].fillna('Unknown'))
        df_work['category_encoded'] = le_category.fit_transform(df_work['category'].fillna('Unknown'))
        df_work['difficulty_encoded'] = le_difficulty.fit_transform(df_work['difficulty_level'].fillna('Unknown'))
        train_mask = df_work['duration_minutes'].notna() & (df_work['duration_minutes'] > 0)
        test_mask = df_work['duration_minutes'].isna() | (df_work['duration_minutes'] == 0)
        if train_mask.sum() > 10: 
            features = ['platform_encoded', 'category_encoded', 'difficulty_encoded', 
                       'title_length', 'has_description', 'desc_length', 'is_free']
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(df_work[train_mask][features], df_work[train_mask]['duration_minutes'])
            predicted_durations = rf_model.predict(df_work[test_mask][features])
            predicted_durations = np.clip(predicted_durations, 5, 480)  # 5 min to 8 hours
            df_work.loc[test_mask, 'duration_minutes'] = predicted_durations
        platform_medians = df_work.groupby('platform')['duration_minutes'].median()
        for platform in df_work['platform'].unique():
            platform_mask = (df_work['platform'] == platform) & (df_work['duration_minutes'].isna())
            if platform_mask.sum() > 0 and platform in platform_medians:
                df_work.loc[platform_mask, 'duration_minutes'] = platform_medians[platform]
        return df_work
    def _impute_difficulty_level_smart(self, df):
        difficulty_keywords = {
            'Beginner': ['intro', 'basic', 'beginner', 'start', 'fundamentals', 'getting started', 
                        'basics', 'first', 'new', 'simple', 'easy', 'introduction', 'overview'],
            'Intermediate': ['intermediate', 'practical', 'applied', 'hands-on', 'real-world', 
                           'project', 'build', 'create', 'develop', 'workshop', 'practice'],
            'Advanced': ['advanced', 'expert', 'master', 'deep', 'complex', 'professional', 
                        'optimization', 'architecture', 'enterprise', 'scaling', 'mastery']
        }
        difficulty_series = df['difficulty_level'].copy()
        for idx, row in df.iterrows():
            if pd.isna(difficulty_series.iloc[idx]) or difficulty_series.iloc[idx] == 'Unknown':
                text_to_analyze = str(row['title']).lower()
                if pd.notna(row['description']):
                    text_to_analyze += ' ' + str(row['description']).lower()
                scores = {}
                for level, keywords in difficulty_keywords.items():
                    scores[level] = sum(1 for keyword in keywords if keyword in text_to_analyze)
                duration = row['duration_minutes']
                if pd.notna(duration) and duration > 0:
                    if duration < 30:
                        scores['Beginner'] += 2
                    elif duration > 240:  # 4 hours
                        scores['Advanced'] += 2
                    else:
                        scores['Intermediate'] += 1
                platform = row['platform']
                if platform == 'Khan Academy':
                    scores['Beginner'] += 1
                elif platform == 'Coursera':
                    scores['Intermediate'] += 1
                elif platform == 'YouTube':
                    scores['Beginner'] += 1
                if row['is_free'] == False and pd.notna(row['price']) and row['price'] > 100:
                    scores['Advanced'] += 1
                if pd.notna(row['category']):
                    category = str(row['category']).lower()
                    if 'advanced' in category or 'master' in category:
                        scores['Advanced'] += 2
                    elif 'intro' in category or 'basic' in category:
                        scores['Beginner'] += 2
                if max(scores.values()) > 0:
                    predicted_level = max(scores.items(), key=lambda x: x[1])[0]
                    difficulty_series.iloc[idx] = predicted_level
                else:
                    difficulty_series.iloc[idx] = 'Intermediate'  # Safe default
        return difficulty_series
    def _impute_language_smart(self, df):
        df_work = df.copy()
        platform_languages = {
            'Khan Academy': 'English',
            'Coursera': 'English',
            'YouTube': 'English',
            'Reddit': 'English',

        }
        for platform, default_lang in platform_languages.items():
            mask = (df_work['platform'] == platform) & (df_work['language'].isna())
            df_work.loc[mask, 'language'] = default_lang
        language_keywords = {
            'Spanish': ['español', 'curso', 'para', 'con', 'en', 'de', 'la', 'el'],
            'French': ['français', 'cours', 'pour', 'avec', 'dans', 'le', 'la', 'de'],
            'German': ['deutsch', 'kurs', 'für', 'mit', 'und', 'der', 'die', 'das'],
            'Portuguese': ['português', 'curso', 'para', 'com', 'em', 'de', 'da', 'do'],
            'Italian': ['italiano', 'corso', 'per', 'con', 'in', 'il', 'la', 'di'],
            'Chinese': ['中文', '课程', '学习', '教程', '中国', '汉语'],
            'Japanese': ['日本語', 'コース', '学習', 'チュートリアル', '日本', '語学'],
            'Korean': ['한국어', '과정', '학습', '튜토리얼', '한국', '어학'],
            'Russian': ['русский', 'курс', 'для', 'с', 'в', 'на', 'по'],
            'Hindi': ['हिंदी', 'कोर्स', 'सीखना', 'ट्यूटोरियल', 'भारत', 'हिन्दी']
        }
        remaining_mask = df_work['language'].isna()
        for idx, row in df_work[remaining_mask].iterrows():
            text_to_check = str(row['title']).lower()
            if pd.notna(row['description']):
                text_to_check += ' ' + str(row['description']).lower()
            detected_lang = None
            max_score = 0
            for lang, keywords in language_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_to_check)
                if score > max_score:
                    max_score = score
                    detected_lang = lang
            df_work.loc[idx, 'language'] = detected_lang if detected_lang else 'English'
        return df_work
    def _impute_category_smart(self, df):
        df_work = df.copy()
        category_keywords = {
            'Programming': ['programming', 'code', 'coding', 'python', 'java', 'javascript', 'html', 'css', 'sql', 'c++', 'php', 'ruby', 'swift', 'kotlin', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'bootstrap', 'jquery', 'api', 'database', 'algorithm', 'data structure', 'software development', 'web development', 'mobile development'],
            'Data Science': ['data science', 'machine learning', 'ai', 'artificial intelligence', 'statistics', 'analytics', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'tensorflow', 'pytorch', 'r programming', 'tableau', 'power bi', 'excel', 'big data', 'hadoop', 'spark', 'data mining', 'predictive modeling', 'neural networks', 'deep learning'],
            'Mathematics': ['mathematics', 'math', 'calculus', 'algebra', 'geometry', 'trigonometry', 'statistics', 'probability', 'linear algebra', 'discrete math', 'number theory', 'differential equations', 'mathematical analysis', 'logic', 'set theory'],
            'Science': ['physics', 'chemistry', 'biology', 'science', 'astronomy', 'geology', 'environmental science', 'neuroscience', 'genetics', 'molecular biology', 'organic chemistry', 'quantum physics', 'biochemistry', 'microbiology', 'ecology', 'zoology', 'botany'],
            'Business': ['business', 'management', 'marketing', 'finance', 'accounting', 'economics', 'entrepreneurship', 'leadership', 'strategy', 'operations', 'project management', 'human resources', 'sales', 'negotiation', 'consulting', 'investment', 'banking', 'trading', 'startup', 'venture capital'],
            'Arts': ['art', 'arts', 'design', 'creative', 'drawing', 'painting', 'photography', 'music', 'writing', 'literature', 'poetry', 'creative writing', 'graphic design', 'illustration', 'sculpture', 'dance', 'theater', 'film', 'animation', 'digital art'],
            'Language': ['language', 'english', 'spanish', 'french', 'german', 'chinese', 'japanese', 'grammar', 'vocabulary', 'pronunciation', 'conversation', 'writing', 'reading', 'listening', 'speaking', 'linguistics', 'translation', 'communication'],
            'Engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical', 'aerospace', 'biomedical', 'computer engineering', 'structural', 'materials', 'thermodynamics', 'circuits', 'systems', 'automation', 'robotics', 'cad', 'solidworks', 'autocad'],
            'Medicine': ['medicine', 'medical', 'health', 'anatomy', 'physiology', 'pharmacology', 'pathology', 'clinical', 'nursing', 'surgery', 'cardiology', 'neurology', 'pediatrics', 'psychiatry', 'radiology', 'emergency medicine', 'public health', 'epidemiology'],
            'Technology': ['technology', 'tech', 'computer', 'digital', 'cyber', 'network', 'cloud', 'devops', 'blockchain', 'cryptocurrency', 'iot', 'internet of things', 'virtual reality', 'augmented reality', 'security', 'cybersecurity', 'infrastructure', 'system administration']
        }
        remaining_mask = df_work['category'].isna()
        for idx, row in df_work[remaining_mask].iterrows():
            text_to_check = str(row['title']).lower()
            if pd.notna(row['description']):
                text_to_check += ' ' + str(row['description']).lower()
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_to_check)
                if score > 0:
                    category_scores[category] = score
            if category_scores:
                predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]
                df_work.loc[idx, 'category'] = predicted_category
            else:
                df_work.loc[idx, 'category'] = 'General'
        return df_work
    def _impute_instructor_smart(self, df):
        df_work = df.copy()
        platform_instructor_patterns = {
            'Khan Academy': 'Khan Academy',
            'YouTube': lambda title: self._extract_youtube_instructor(title),
            'Reddit': 'Community Author',
            'Coursera': 'Course Instructor',
        }
        remaining_mask = df_work['instructor'].isna()
        for idx, row in df_work[remaining_mask].iterrows():
            platform = row['platform']
            if platform in platform_instructor_patterns:
                pattern = platform_instructor_patterns[platform]
                if callable(pattern):
                    instructor = pattern(row['title'])
                else:
                    instructor = pattern
                df_work.loc[idx, 'instructor'] = instructor
            else:
                df_work.loc[idx, 'instructor'] = 'Unknown Instructor'
        return df_work
    def _extract_youtube_instructor(self, title):
        title_str = str(title)
        patterns = [
            r'by\s+([A-Za-z\s]+)',
            r'with\s+([A-Za-z\s]+)',
            r'([A-Za-z\s]+)\s+teaches',
            r'([A-Za-z\s]+)\s+explains'
        ]
        for pattern in patterns:
            match = re.search(pattern, title_str, re.IGNORECASE)
            if match:
                instructor = match.group(1).strip()
                if len(instructor) > 2 and len(instructor) < 50:
                    return instructor
        return 'YouTube Creator'
    def _fix_price_consistency(self, df):
        df_work = df.copy()
        free_with_price = (df_work['is_free'] == True) & (df_work['price'] > 0)
        df_work.loc[free_with_price, 'price'] = 0.0
        paid_without_price = (df_work['is_free'] == False) & (df_work['price'] == 0)
        platform_avg_prices = df_work.groupby('platform')['price'].mean()
        for idx, row in df_work[paid_without_price].iterrows():
            platform = row['platform']
            if platform in platform_avg_prices and platform_avg_prices[platform] > 0:
                df_work.loc[idx, 'price'] = platform_avg_prices[platform]
            else:
                df_work.loc[idx, 'price'] = 49.99  
        return df_work
    def _log_imputation_improvements(self, imputed_df):
        original_df = self.courses_df
        improvements = {}
        fields_to_check = ['duration_minutes', 'difficulty_level', 'language', 'category', 'instructor']
        for field in fields_to_check:
            if field in original_df.columns and field in imputed_df.columns:
                original_completeness = 1 - (original_df[field].isnull().sum() / len(original_df))
                new_completeness = 1 - (imputed_df[field].isnull().sum() / len(imputed_df))
                improvements[field] = {
                    'original': original_completeness,
                    'improved': new_completeness,
                    'improvement': new_completeness - original_completeness
                }
        self.logger.info("=== IMPUTATION IMPROVEMENTS ===")
        for field, stats in improvements.items():
            self.logger.info(f"{field}: {stats['original']:.1%} → {stats['improved']:.1%} (+{stats['improvement']:.1%})")
    def create_advanced_engagement_metrics(self):
     if not hasattr(self, 'engagement_pivot') or len(self.engagement_pivot) == 0:
        self.logger.warning("No engagement data available for advanced metrics")
        return pd.DataFrame()

     enhanced_engagement = self.engagement_pivot.copy()

     # Compute engagement rate if views are available
     if 'views' in enhanced_engagement.columns:
        enhanced_engagement['engagement_rate'] = (
            enhanced_engagement.get('likes', 0) +
            enhanced_engagement.get('comments', 0)
        ) / enhanced_engagement['views'].replace(0, 1)

     # Weighted engagement score
     metrics = ['views', 'likes', 'comments', 'enrollments']
     weights = [0.4, 0.3, 0.2, 0.1]
     enhanced_engagement['engagement_score'] = 0
     for metric, weight in zip(metrics, weights):
        if metric in enhanced_engagement.columns:
            enhanced_engagement['engagement_score'] += weight * enhanced_engagement[metric]

     # Normalize engagement scores per platform if course data is available
     if hasattr(self, 'courses_df'):
        course_platform = self.courses_df[['course_id', 'platform']].set_index('course_id')
        if course_platform.empty:
            self.logger.warning("Course-platform mapping is empty. Skipping normalization.")
        else:
            enhanced_engagement = enhanced_engagement.join(course_platform, on='course_id')
            platforms = enhanced_engagement['platform'].dropna().unique()
            self.logger.info(f"Platforms found: {platforms}")
            for platform in platforms:
                platform_mask = enhanced_engagement['platform'] == platform
                max_score = enhanced_engagement.loc[platform_mask, 'engagement_score'].max()
                self.logger.info(f"Max engagement score for platform '{platform}': {max_score}")
                if max_score > 0:
                    enhanced_engagement.loc[platform_mask, 'engagement_score_normalized'] = (
                        enhanced_engagement.loc[platform_mask, 'engagement_score'] / max_score
                    )
    
     # Ensure the column exists before using it
     if 'engagement_score_normalized' not in enhanced_engagement.columns:
        self.logger.warning("'engagement_score_normalized' not found. Filling with zeros.")
        enhanced_engagement['engagement_score_normalized'] = 0

     # Assign category based on normalized score
     enhanced_engagement['engagement_category'] = pd.cut(
         enhanced_engagement['engagement_score_normalized'].fillna(0),
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

     self.enhanced_engagement = enhanced_engagement
     return enhanced_engagement

    def comprehensive_data_enhancement(self):
     self.logger.info("Starting comprehensive data enhancement...")
     self._load_enhanced_data()
     if not hasattr(self, 'courses_df'):
        self.logger.error("courses_df not loaded. Check _load_enhanced_data.")
        raise ValueError("courses_df is missing after data load.")
     enhanced_df = self.enhance_descriptions()
     self.courses_df = enhanced_df  
     imputed_df = self.smart_data_imputation()
     enhanced_engagement = self.create_advanced_engagement_metrics()
     if hasattr(self, 'enhanced_engagement'):
        merged_enhanced = pd.merge(
            imputed_df,
            enhanced_engagement,
            on='course_id',
            how='left'
        )
        self.final_enhanced_df = merged_enhanced
     else:
        self.final_enhanced_df = imputed_df
     final_quality = self._validate_enhanced_data()
     self.logger.info("Comprehensive data enhancement completed!")
     return self.final_enhanced_df, final_quality
    def _validate_enhanced_data(self):
        if not hasattr(self, 'final_enhanced_df'):
            return {}
        df = self.final_enhanced_df
        validation_results = {
            'completeness_improvement': {},
            'consistency_checks': {},
            'quality_scores': {}
        }
        critical_fields = ['title', 'instructor', 'duration_minutes', 'difficulty_level', 'description']
        for field in critical_fields:
            if field in df.columns:
                completeness = 1 - (df[field].isnull().sum() / len(df))
                validation_results['completeness_improvement'][field] = completeness
        consistency_checks = {
            'price_consistency': (df['is_free'] == (df['price'] == 0)).mean(),
            'duration_validity': ((df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 1440)).mean(),
            'category_coverage': df['category'].notna().mean()
        }
        validation_results['consistency_checks'] = consistency_checks
        validation_results['quality_scores']['overall'] = np.mean([
            np.mean(list(validation_results['completeness_improvement'].values())),
            np.mean(list(consistency_checks.values()))
        ])
        return validation_results
    def save_enhanced_data(self):
     try:
        with self.app.app_context():
            if hasattr(self, 'final_enhanced_df'):
                self.final_enhanced_df.to_sql(
                    'courses_enhanced_analysis',
                    self.db.engine,
                    if_exists='replace',
                    index=False,
                    method='multi'
                )
                self.logger.info("Enhanced course data saved to 'courses_enhanced_analysis'")
            if hasattr(self, 'enhanced_engagement'):
                self.enhanced_engagement.to_sql(
                    'engagement_enhanced_analysis',
                    self.db.engine,
                    if_exists='replace',
                    index=False,
                    method='multi'
                )
                self.logger.info("Enhanced engagement data saved to 'engagement_enhanced_analysis'")
            quality_report = self.comprehensive_quality_assessment()
            quality_df = pd.DataFrame([{
                'assessment_date': datetime.now(),
                'total_courses': len(self.courses_df),
                'overall_quality_score': np.mean([
                 np.mean([v['score'] for v in quality_report['completeness'].values() if isinstance(v, dict)]),
                 quality_report['consistency']['overall_score'],
                 quality_report['validity']['overall_validity']]),
                'completeness_score': np.mean([
                    v['score'] for v in quality_report['completeness'].values() 
                    if isinstance(v, dict) and 'score' in v
                ]),
                'consistency_score': quality_report['consistency']['overall_score'],
                'validity_score': quality_report['validity']['overall_validity']
            }])
            quality_df.to_sql(
                'quality_assessment_history',
                self.db.engine,
                if_exists='append',
                index=False
            )
            self.logger.info("Quality assessment saved to 'quality_assessment_history'")
     except Exception as e:
        self.logger.error(f"Error saving enhanced data: {e}")
        raise
def run_comprehensive_enhancement(app):
    enhancer = AdvancedEduDataQuality(app)
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA ENHANCEMENT PIPELINE")
    print("="*60)
    enhanced_df, quality_results = enhancer.comprehensive_data_enhancement()
    enhancer.save_enhanced_data()
    
    print(f"Total courses processed: {len(enhanced_df)}")
    print(f"Overall quality score: {quality_results['quality_scores']['overall']:.3f}")
    
    for field, score in quality_results['completeness_improvement'].items():
        print(f"  {field}: {score:.1%}")
    for check, score in quality_results['consistency_checks'].items():
        print(f"  {check}: {score:.1%}")
    
    return enhanced_df, quality_results

if __name__ == "__main__":
    pass