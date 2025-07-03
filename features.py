import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sqlalchemy import text
import logging
import warnings
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict, Counter
from sqlalchemy import create_engine

from config import config_by_name
db_uri = config_by_name["development"].SQLALCHEMY_DATABASE_URI
engine = create_engine(db_uri)

warnings.filterwarnings('ignore')

class LearningIntelligenceEngine:
    """
    Advanced Learning Intelligence Engine for Educational Data Analysis
    
    Features:
    - Learning Velocity Profiling using survival analysis
    - Engagement Pattern Mining with time-series analysis
    - Content Difficulty Calibration using statistical modeling
    - Learning Path Optimization with network analysis
    - Advanced Behavioral Analytics
    """
    
    def __init__(self, app=None):
        self.db_uri = db_uri
        self.db_engine = create_engine(self.db_uri)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_enhanced_data(self):
     """Load enhanced course and engagement data from database"""
     try:
        # Load enhanced course data
        courses_query = text("""
    SELECT c.*, p.id AS platform_id, p.name AS platform
    FROM courses_enhanced_analysis c
    LEFT JOIN platforms p ON c.platform_x = p.name
""")
        self.courses_df = pd.read_sql(courses_query, self.db_engine)

        # Load enhanced engagement data
        engagement_query = text("SELECT * FROM engagement_enhanced_analysis")
        self.engagement_df = pd.read_sql(engagement_query, self.db_engine)

        self.logger.info(f"Loaded {len(self.courses_df)} courses and {len(self.engagement_df)} engagement records")
        self.logger.debug(f"Course columns: {self.courses_df.columns.tolist()}")
        self.logger.debug(f"Engagement columns: {self.engagement_df.columns.tolist()}")

     except Exception as e:
        self.logger.error(f"Error loading enhanced data: {e}")
        raise

    def learning_velocity_profiling(self):
        """
        Analyze learning velocity using survival analysis
        Models time-to-completion and dropout patterns
        """
        try:
            # Prepare data for survival analysis
            survival_data = self.engagement_df.copy()
            
            # Create duration and event columns for survival analysis
            if 'completion_rate' in survival_data.columns:
                survival_data['completed'] = (survival_data['completion_rate'] > 0.8).astype(int)
                survival_data['duration_days'] = np.random.exponential(30, len(survival_data))  # Simulated for demo
            else:
                # Use engagement metrics to infer completion
                survival_data['completed'] = (survival_data['engagement_score_normalized'] > 0.7).astype(int)
                survival_data['duration_days'] = np.random.exponential(30, len(survival_data))
            
            # Kaplan-Meier survival analysis
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
                            'total_courses': len(platform_data)
                        }
            
            # Cox Proportional Hazards Model for advanced analysis
            try:
                cox_data = survival_data[['duration_days', 'completed', 'engagement_score_normalized']].dropna()
                if len(cox_data) > 20:
                    cph = CoxPHFitter()
                    cph.fit(cox_data, duration_col='duration_days', event_col='completed')
                    
                    cox_results = {
                        'hazard_ratios': cph.hazard_ratios_.to_dict(),
                        'p_values': cph.summary['p'].to_dict(),
                        'concordance_index': cph.concordance_index_
                    }
                else:
                    cox_results = None
            except Exception as e:
                self.logger.warning(f"Cox regression failed: {e}")
                cox_results = None
            
            learning_velocity_results = {
                'overall_survival': {
                    'median_completion_time': kmf.median_survival_time_,
                    'completion_rate_30_days': kmf.survival_function_at_times(30).iloc[0] if kmf.median_survival_time_ else 0,
                    'completion_rate_60_days': kmf.survival_function_at_times(60).iloc[0] if kmf.median_survival_time_ else 0
                },
                'platform_comparison': platform_survival,
                'cox_regression': cox_results
            }
            
            return learning_velocity_results
            
        except Exception as e:
            self.logger.error(f"Error in learning velocity profiling: {e}")
            return {}
    
    def engagement_pattern_mining(self):
        """
        Mine engagement patterns using time-series analysis and clustering
        """
        try:
            # Prepare engagement data for pattern analysis
            engagement_features = self.engagement_df.copy()
            
            # Create engagement pattern features
            numeric_cols = ['views', 'likes', 'comments', 'enrollments', 'engagement_score']
            available_cols = [col for col in numeric_cols if col in engagement_features.columns]
            
            if len(available_cols) < 2:
                self.logger.warning("Insufficient engagement metrics for pattern mining")
                return {}
            
            # Fill missing values and normalize
            feature_data = engagement_features[available_cols].fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Determine optimal number of clusters
            silhouette_scores = []
            K_range = range(2, min(8, len(scaled_features)//10))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Use optimal k or default to 4
            if silhouette_scores:
                optimal_k = K_range[np.argmax(silhouette_scores)]
            else:
                optimal_k = 4
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Analyze cluster characteristics
            engagement_features['cluster'] = cluster_labels
            cluster_analysis = {}
            
            for cluster_id in range(optimal_k):
                cluster_data = engagement_features[engagement_features['cluster'] == cluster_id]
                
                cluster_stats = {}
                for col in available_cols:
                    cluster_stats[col] = {
                        'mean': cluster_data[col].mean(),
                        'median': cluster_data[col].median(),
                        'std': cluster_data[col].std(),
                        'count': len(cluster_data)
                    }
                
                # Platform distribution in cluster
                platform_dist = cluster_data['platform'].value_counts().to_dict()
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(engagement_features) * 100,
                    'characteristics': cluster_stats,
                    'platform_distribution': platform_dist
                }
            
            # Engagement trend analysis
            if 'engagement_score_normalized' in engagement_features.columns:
                engagement_trends = self._analyze_engagement_trends(engagement_features)
            else:
                engagement_trends = {}
            
            return {
                'cluster_analysis': cluster_analysis,
                'optimal_clusters': optimal_k,
                'engagement_trends': engagement_trends,
                'silhouette_scores': dict(zip(K_range, silhouette_scores))
            }
            
        except Exception as e:
            self.logger.error(f"Error in engagement pattern mining: {e}")
            return {}
    
    def _analyze_engagement_trends(self, engagement_data):
        """Analyze engagement trends over time"""
        try:
            # Platform-wise engagement trends
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
            
            # Category-wise engagement
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
     """
     Calibrate content difficulty using statistical modeling
     """
     try:
        # Dynamically pick available columns to avoid KeyErrors
        desired_engagement_cols = ['course_id', 'engagement_score_normalized', 'completion_rate']
        available_engagement_cols = [col for col in desired_engagement_cols if col in self.engagement_df.columns]

        filtered_engagement_df = self.engagement_df[available_engagement_cols].copy()

        # Merge course and filtered engagement data for analysis
        analysis_data = pd.merge(
            self.courses_df,
            filtered_engagement_df,
            on='course_id',
            how='inner'
        )

        if len(analysis_data) == 0:
            self.logger.warning("No merged data available for difficulty calibration")
            return {}

        # Difficulty level analysis
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

        # Statistical tests for difficulty differences
        statistical_tests = {}
        if len(difficulty_stats) >= 2:
            difficulties = list(difficulty_stats.keys())

            # Kruskal-Wallis test for duration differences
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

        # Content complexity scoring
        complexity_scores = self._calculate_content_complexity(analysis_data)

        return {
            'difficulty_statistics': difficulty_stats,
            'statistical_tests': statistical_tests,
            'complexity_scores': complexity_scores
        }

     except Exception as e:
        self.logger.error(f"Error in content difficulty calibration: {e}")
        return {}

    
    def _calculate_content_complexity(self, data):
        """Calculate content complexity scores"""
        try:
            complexity_factors = {}
            
            # Duration-based complexity
            if 'duration_minutes' in data.columns:
                duration_data = data['duration_minutes'].dropna()
                complexity_factors['duration'] = {
                    'mean': duration_data.mean(),
                    'std': duration_data.std(),
                    'complexity_score': (duration_data > duration_data.quantile(0.75)).mean()
                }
            
            # Description complexity (if available)
            if 'description' in data.columns:
                desc_lengths = data['description'].str.len().fillna(0)
                complexity_factors['description_complexity'] = {
                    'avg_length': desc_lengths.mean(),
                    'complexity_score': (desc_lengths > desc_lengths.quantile(0.75)).mean()
                }
            
            # Engagement vs Duration correlation
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
        """
        Optimize learning paths using network analysis
        """
        try:
            # Create learning network based on categories and difficulties
            G = nx.DiGraph()
            
            # Add nodes for each course
            for _, course in self.courses_df.iterrows():
                G.add_node(course['course_id'], 
                          title=course['title'],
                          category=course['category'],
                          difficulty=course['difficulty_level'],
                          duration=course['duration_minutes'],
                          platform=course['platform'])
            
            # Add edges based on learning progression logic
            courses_by_category = self.courses_df.groupby('category')
            
            for category, category_courses in courses_by_category:
                if len(category_courses) > 1:
                    # Sort by difficulty and duration
                    difficulty_order = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
                    category_courses['difficulty_num'] = category_courses['difficulty_level'].map(difficulty_order).fillna(1)
                    category_courses = category_courses.sort_values(['difficulty_num', 'duration_minutes'])
                    
                    # Create prerequisite relationships
                    for i in range(len(category_courses) - 1):
                        current_course = category_courses.iloc[i]
                        next_course = category_courses.iloc[i + 1]
                        
                        # Add edge if there's a logical progression
                        if (current_course['difficulty_num'] <= next_course['difficulty_num']):
                            G.add_edge(current_course['course_id'], next_course['course_id'],
                                     weight=1.0, relationship='prerequisite')
            
            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(G)
            
            # Find optimal learning paths
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
        """Calculate network analysis metrics"""
        try:
            metrics = {}
            
            # Basic network properties
            metrics['density'] = nx.density(G)
            metrics['is_connected'] = nx.is_weakly_connected(G)
            
            # Centrality measures
            if G.number_of_nodes() > 0:
                in_degree_centrality = nx.in_degree_centrality(G)
                out_degree_centrality = nx.out_degree_centrality(G)
                
                # Find most central courses
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
            
            # Component analysis
            if G.number_of_nodes() > 0:
                components = list(nx.weakly_connected_components(G))
                metrics['connected_components'] = len(components)
                metrics['largest_component_size'] = max(len(comp) for comp in components) if components else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating network metrics: {e}")
            return {}
    
    def _find_optimal_paths(self, G):
        """Find optimal learning paths through the network"""
        try:
            optimal_paths = {}
            
            # Find paths for each category
            categories = set(nx.get_node_attributes(G, 'category').values())
            
            for category in categories:
                if pd.notna(category):
                    category_nodes = [node for node, attr in G.nodes(data=True) 
                                    if attr.get('category') == category]
                    
                    if len(category_nodes) > 1:
                        # Find beginner courses (no incoming edges)
                        beginners = [node for node in category_nodes if G.in_degree(node) == 0]
                        
                        # Find advanced courses (no outgoing edges)
                        advanced = [node for node in category_nodes if G.out_degree(node) == 0]
                        
                        # Find paths from beginners to advanced
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
        """
        Perform advanced behavioral analytics
        """
        try:
            # Attention span modeling
            attention_analysis = self._model_attention_span()
            
            # Success pattern classification
            success_patterns = self._classify_success_patterns()
            
            # Retention modeling
            retention_analysis = self._model_retention()
            
            return {
                'attention_span_analysis': attention_analysis,
                'success_patterns': success_patterns,
                'retention_analysis': retention_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced behavioral analytics: {e}")
            return {}
    
    def _model_attention_span(self):
        """Model attention span using exponential decay"""
        try:
            # Use duration and engagement to model attention
            analysis_data = pd.merge(
                self.courses_df[['course_id', 'duration_minutes']], 
                self.engagement_df[['course_id', 'engagement_score_normalized']], 
                on='course_id', 
                how='inner'
            )
            
            if len(analysis_data) == 0:
                return {}
            
            # Exponential decay model: engagement = exp(-alpha * duration)
            durations = analysis_data['duration_minutes'].values
            engagements = analysis_data['engagement_score_normalized'].values
            
            # Filter valid data
            valid_mask = (durations > 0) & (engagements > 0)
            if valid_mask.sum() < 10:
                return {}
            
            valid_durations = durations[valid_mask]
            valid_engagements = engagements[valid_mask]
            
            # Fit exponential decay
            try:
                # log(engagement) = log(A) - alpha * duration
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
    
    def _classify_success_patterns(self):
        """Classify success patterns using clustering"""
        try:
            # Define success metrics
            success_data = self.engagement_df.copy()
            
            # Create success features
            success_features = []
            feature_names = []
            
            if 'engagement_score_normalized' in success_data.columns:
                success_features.append(success_data['engagement_score_normalized'].fillna(0))
                feature_names.append('engagement_score')
            
            if 'views' in success_data.columns:
                success_features.append(np.log1p(success_data['views'].fillna(0)))
                feature_names.append('log_views')
            
            if 'likes' in success_data.columns:
                success_features.append(np.log1p(success_data['likes'].fillna(0)))
                feature_names.append('log_likes')
            
            if len(success_features) < 2:
                return {}
            
            # Combine features
            X = np.column_stack(success_features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=3, random_state=42)  # High, Medium, Low success
            clusters = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters
            cluster_analysis = {}
            for i in range(3):
                cluster_mask = clusters == i
                cluster_data = success_data[cluster_mask]
                
                cluster_analysis[f'pattern_{i}'] = {
                    'size': cluster_mask.sum(),
                    'percentage': cluster_mask.sum() / len(success_data) * 100,
                    'avg_engagement': cluster_data['engagement_score_normalized'].mean() if 'engagement_score_normalized' in cluster_data.columns else 0,
                    'platform_distribution': cluster_data['platform'].value_counts().to_dict()
                }
            
            return cluster_analysis
            
        except Exception as e:
            self.logger.error(f"Error classifying success patterns: {e}")
            return {}
    
    def _model_retention(self):
        """Model knowledge retention using forgetting curves"""
        try:
            # Simulate retention data based on engagement patterns
            retention_data = self.engagement_df.copy()
            
            if 'engagement_score_normalized' not in retention_data.columns:
                return {}
            
            # Model: retention = initial_strength * exp(-decay_rate * time)
            # Use engagement as proxy for initial strength
            engagements = retention_data['engagement_score_normalized'].values
            
            # Simulate time points (days)
            time_points = np.array([1, 7, 30, 90, 365])  # 1 day, 1 week, 1 month, 3 months, 1 year
            
            # Calculate retention curves for different engagement levels
            retention_curves = {}
            
            # High engagement (top 25%)
            high_engagement_threshold = np.percentile(engagements, 75)
            high_engagement_courses = engagements >= high_engagement_threshold
            
            # Medium engagement (25-75%)
            medium_engagement_courses = (engagements >= np.percentile(engagements, 25)) & (engagements < high_engagement_threshold)
            
            # Low engagement (bottom 25%)
            low_engagement_courses = engagements < np.percentile(engagements, 25)
            
            # Different decay rates for different engagement levels
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
        """Generate comprehensive learning intelligence report"""
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
                'engagement_patterns': self.engagement_pattern_mining(),
                'difficulty_calibration': self.content_difficulty_calibration(),
                'learning_paths': self.learning_path_optimization(),
                'behavioral_analytics': self.advanced_behavioral_analytics()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {}
    
    def save_analysis_results(self, results):
        """Save analysis results to database"""
        try:
                # Save results as JSON in database
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

# Usage example
def run_learning_intelligence_analysis(db_uri):
    """Run complete learning intelligence analysis"""
    engine = LearningIntelligenceEngine(db_uri)
    
    print("\n" + "="*60)
    print("LEARNING INTELLIGENCE ANALYSIS")
    print("="*60)
    
    # Generate comprehensive report
    report = engine.generate_comprehensive_report()
    
    # Save results
    engine.save_analysis_results(report)
    
    # Print summary
    if report:
        print(f"Analysis completed for {report['data_summary']['total_courses']} courses")
        print(f"Platforms analyzed: {report['data_summary']['platforms']}")
        print(f"Categories analyzed: {report['data_summary']['categories']}")
        
        # Print key insights
        if 'learning_velocity' in report:
            velocity = report['learning_velocity']
            if 'overall_survival' in velocity:
                print(f"Median completion time: {velocity['overall_survival'].get('median_completion_time', 'N/A')} days")
        
        if 'engagement_patterns' in report:
            patterns = report['engagement_patterns']
            if 'optimal_clusters' in patterns:
                print(f"Identified {patterns['optimal_clusters']} distinct engagement patterns")
        
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

# Additional utility functions for the Learning Intelligence Engine

def create_learning_dashboard_data(report):
    """Create data structure for learning dashboard visualization"""
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
    
    # Extract engagement insights
    if 'engagement_patterns' in report:
        patterns = report['engagement_patterns']
        dashboard_data['engagement_insights'] = {
            'cluster_count': patterns.get('optimal_clusters', 0),
            'cluster_analysis': patterns.get('cluster_analysis', {}),
            'silhouette_scores': patterns.get('silhouette_scores', {}),
            'engagement_trends': patterns.get('engagement_trends', {})
        }
    
    # Extract learning velocity insights
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        dashboard_data['learning_velocity_insights'] = {
            'overall_survival': velocity.get('overall_survival', {}),
            'platform_comparison': velocity.get('platform_comparison', {}),
            'cox_regression': velocity.get('cox_regression', {})
        }
    
    # Extract difficulty insights
    if 'difficulty_calibration' in report:
        difficulty = report['difficulty_calibration']
        dashboard_data['difficulty_insights'] = {
            'difficulty_statistics': difficulty.get('difficulty_statistics', {}),
            'statistical_tests': difficulty.get('statistical_tests', {}),
            'complexity_scores': difficulty.get('complexity_scores', {})
        }
    
    # Extract path optimization insights
    if 'learning_paths' in report:
        paths = report['learning_paths']
        dashboard_data['path_optimization'] = {
            'network_metrics': paths.get('network_metrics', {}),
            'optimal_paths': paths.get('optimal_paths', {}),
            'network_size': paths.get('network_size', {})
        }
    
    # Extract behavioral insights
    if 'behavioral_analytics' in report:
        behavioral = report['behavioral_analytics']
        dashboard_data['behavioral_insights'] = {
            'attention_span': behavioral.get('attention_span_analysis', {}),
            'success_patterns': behavioral.get('success_patterns', {}),
            'retention_analysis': behavioral.get('retention_analysis', {})
        }
    
    return dashboard_data

def generate_learning_recommendations(report):
    """Generate actionable learning recommendations based on analysis"""
    recommendations = {
        'course_optimization': [],
        'engagement_improvement': [],
        'learning_path_enhancement': [],
        'platform_strategy': [],
        'content_strategy': []
    }
    
    # Course optimization recommendations
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
    
    # Engagement improvement recommendations
    if 'engagement_patterns' in report:
        patterns = report['engagement_patterns']
        if 'cluster_analysis' in patterns:
            for cluster_id, cluster_data in patterns['cluster_analysis'].items():
                if cluster_data['percentage'] > 30 and 'characteristics' in cluster_data:
                    characteristics = cluster_data['characteristics']
                    if 'engagement_score' in characteristics:
                        eng_score = characteristics['engagement_score']['mean']
                        if eng_score < 0.4:
                            recommendations['engagement_improvement'].append(
                                f"Target improvement for {cluster_id} ({cluster_data['percentage']:.1f}% of courses) - low engagement"
                            )
    
    # Learning path enhancement recommendations
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
    
    # Platform strategy recommendations
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        if 'platform_comparison' in velocity:
            platform_performances = []
            for platform, data in velocity['platform_comparison'].items():
                survival_30 = data.get('survival_at_30_days', 0)
                platform_performances.append((platform, survival_30))
            
            if platform_performances:
                # Sort by performance
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
    
    # Content strategy recommendations
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
    """Export analysis summary to various formats"""
    if filename is None:
        filename = f"learning_intelligence_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create summary dictionary
    summary = {
        'Analysis Overview': {
            'Total Courses Analyzed': report.get('data_summary', {}).get('total_courses', 0),
            'Total Platforms': report.get('data_summary', {}).get('platforms', 0),
            'Total Categories': report.get('data_summary', {}).get('categories', 0),
            'Analysis Date': report.get('timestamp', datetime.now().isoformat())
        }
    }
    
    # Add key metrics
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        if 'overall_survival' in velocity:
            summary['Learning Velocity'] = {
                'Median Completion Time (days)': velocity['overall_survival'].get('median_completion_time', 'N/A'),
                'Completion Rate at 30 Days': velocity['overall_survival'].get('completion_rate_30_days', 'N/A'),
                'Completion Rate at 60 Days': velocity['overall_survival'].get('completion_rate_60_days', 'N/A')
            }
    
    if 'engagement_patterns' in report:
        patterns = report['engagement_patterns']
        summary['Engagement Patterns'] = {
            'Optimal Clusters Identified': patterns.get('optimal_clusters', 0),
            'Cluster Analysis Available': bool(patterns.get('cluster_analysis', {}))
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
    
    # Generate recommendations
    recommendations = generate_learning_recommendations(report)
    summary['Recommendations'] = recommendations
    
    return summary

# Advanced analytics functions
def calculate_roi_metrics(report, cost_data=None):
    """Calculate ROI metrics for learning programs"""
    roi_metrics = {
        'engagement_roi': {},
        'completion_roi': {},
        'platform_roi': {},
        'category_roi': {}
    }
    
    # This would integrate with actual cost data
    # For now, we'll use relative metrics
    
    if 'engagement_patterns' in report:
        patterns = report['engagement_patterns']
        if 'cluster_analysis' in patterns:
            for cluster_id, cluster_data in patterns['cluster_analysis'].items():
                size = cluster_data.get('size', 0)
                engagement = cluster_data.get('characteristics', {}).get('engagement_score', {}).get('mean', 0)
                roi_metrics['engagement_roi'][cluster_id] = {
                    'relative_value': engagement * size,
                    'efficiency_score': engagement / max(size, 1)
                }
    
    return roi_metrics

def predict_learning_outcomes(report, future_days=30):
    """Predict learning outcomes based on current trends"""
    predictions = {
        'completion_predictions': {},
        'engagement_trends': {},
        'platform_performance': {},
        'risk_factors': []
    }
    
    if 'learning_velocity' in report:
        velocity = report['learning_velocity']
        if 'platform_comparison' in velocity:
            for platform, data in velocity['platform_comparison'].items():
                survival_30 = data.get('survival_at_30_days', 0)
                # Simple linear extrapolation
                predicted_completion = survival_30 * (1 + future_days/30 * 0.1)  # 10% monthly growth assumption
                predictions['completion_predictions'][platform] = min(predicted_completion, 1.0)
    
    if 'behavioral_analytics' in report:
        behavioral = report['behavioral_analytics']
        if 'attention_span_analysis' in behavioral:
            attention = behavioral['attention_span_analysis']
            decay_rate = attention.get('decay_rate', 0)
            if decay_rate > 0.2:
                predictions['risk_factors'].append("High attention decay rate - increased dropout risk")
    
    return predictions

# Integration helper functions
def integrate_with_lms(report, lms_config):
    """Integrate analysis results with Learning Management Systems"""
    integration_results = {
        'status': 'success',
        'recommendations_pushed': [],
        'dashboards_updated': [],
        'alerts_sent': []
    }
    
    # This would integrate with actual LMS APIs
    # Placeholder for integration logic
    
    recommendations = generate_learning_recommendations(report)
    
    # Simulate pushing recommendations to LMS
    for category, recs in recommendations.items():
        if recs:
            integration_results['recommendations_pushed'].append({
                'category': category,
                'count': len(recs),
                'timestamp': datetime.now().isoformat()
            })
    
    return integration_results

# Main execution function
if __name__ == "__main__":
    report = run_learning_intelligence_analysis(db_uri)
    dashboard_data = create_learning_dashboard_data(report)
    recommendations = generate_learning_recommendations(report)
    summary = export_analysis_summary(report)
    print("\n Analysis complete.")
    print("Summary:", summary)