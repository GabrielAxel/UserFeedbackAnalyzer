#!/usr/bin/env python3
"""
User Feedback Analysis Tool

Processes CSV files containing user feedback to:
- Categorize themes using NLP
- Identify feature requests and pain points
- Generate actionable insights for product management

Required columns in CSV: 'feedback' or 'text' or 'comment'
Optional columns: 'user_id', 'timestamp', 'rating', 'category'
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import argparse
from datetime import datetime

# NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
except ImportError:
    print("NLTK not found. Install with: pip install nltk")
    exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Scikit-learn not found. Install with: pip install scikit-learn")
    exit(1)

try:
    from textblob import TextBlob
except ImportError:
    print("TextBlob not found. Install with: pip install textblob")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
except ImportError:
    print("Visualization libraries not found. Install with: pip install matplotlib seaborn")
    plt = None
    sns = None


class FeedbackAnalyzer:
    """Main class for analyzing user feedback from CSV files."""
    
    def __init__(self, csv_path: str, text_column: str = None):
        """
        Initialize the feedback analyzer.
        
        Args:
            csv_path: Path to CSV file containing feedback
            text_column: Name of column containing feedback text
        """
        self.csv_path = Path(csv_path)
        self.text_column = text_column
        self.df = None
        self.processed_texts = []
        self.themes = {}
        self.insights = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Feature request keywords
        self.feature_keywords = [
            'feature', 'add', 'would like', 'wish', 'want', 'need', 'should',
            'could', 'enhancement', 'improvement', 'suggest', 'request',
            'integrate', 'implement', 'support', 'include', 'option'
        ]
        
        # Pain point keywords
        self.pain_keywords = [
            'bug', 'error', 'issue', 'problem', 'difficult', 'hard', 'slow',
            'confusing', 'annoying', 'frustrating', 'broken', 'fail', 'crash',
            'freeze', 'hang', 'stuck', 'terrible', 'awful', 'hate', 'worst'
        ]
    
    def _download_nltk_data(self):
        """Download required NLTK datasets."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for dataset in required_data:
            try:
                nltk.download(dataset, quiet=True)
            except Exception as e:
                self.logger.warning(f"Could not download {dataset}: {e}")
    
    def load_data(self) -> None:
        """Load and validate CSV data."""
        self.df, self.text_column = FeedbackProcessor.load_csv(self.csv_path, self.text_column)
        self.logger.info(f"Loaded {len(self.df)} feedback entries from {self.csv_path}")
        self.logger.info(f"Using text column: {self.text_column}")
    
    def preprocess_texts(self) -> None:
        """Preprocess all feedback texts."""
        processor = TextProcessor(self.lemmatizer, self.stop_words)
        
        self.logger.info("Preprocessing feedback texts...")
        self.df['cleaned_text'] = self.df[self.text_column].apply(processor.clean_text)
        self.df['processed_text'] = self.df[self.text_column].apply(processor.preprocess_text)
        
        # Filter out empty processed texts
        self.df = self.df[self.df['processed_text'].str.strip() != '']
        self.processed_texts = self.df['processed_text'].tolist()
        
        self.logger.info(f"Processed {len(self.processed_texts)} feedback entries")
    
    def analyze_sentiment(self) -> None:
        """Analyze sentiment of feedback."""
        self.logger.info("Analyzing sentiment...")
        
        sentiments = []
        polarities = []
        
        for text in self.df[self.text_column]:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            sentiments.append(sentiment)
            polarities.append(polarity)
        
        self.df['sentiment'] = sentiments
        self.df['polarity'] = polarities
        
        sentiment_counts = self.df['sentiment'].value_counts()
        self.insights['sentiment_distribution'] = sentiment_counts.to_dict()
        
        self.logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
    
    def extract_themes(self, num_themes: int = 5, method: str = 'lda') -> None:
        """Extract themes using topic modeling."""
        self.logger.info(f"Extracting {num_themes} themes using {method.upper()}...")
        
        if len(self.processed_texts) < num_themes:
            self.logger.warning(f"Not enough texts ({len(self.processed_texts)}) for {num_themes} themes")
            num_themes = max(1, len(self.processed_texts) // 2)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        if method.lower() == 'lda':
            # Latent Dirichlet Allocation
            lda = LatentDirichletAllocation(
                n_components=num_themes,
                random_state=42,
                max_iter=100
            )
            lda.fit(tfidf_matrix)
            
            # Extract top words for each theme
            themes = {}
            for theme_idx, theme in enumerate(lda.components_):
                top_words_idx = theme.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                themes[f'Theme_{theme_idx + 1}'] = {
                    'keywords': top_words,
                    'weight': theme.max()
                }
            
            # Assign themes to documents
            doc_topic_probs = lda.transform(tfidf_matrix)
            self.df['primary_theme'] = [f'Theme_{np.argmax(probs) + 1}' for probs in doc_topic_probs]
            self.df['theme_confidence'] = [np.max(probs) for probs in doc_topic_probs]
        
        elif method.lower() == 'kmeans':
            # K-means clustering
            kmeans = KMeans(n_clusters=num_themes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extract representative words for each cluster
            themes = {}
            for cluster_idx in range(num_themes):
                cluster_center = kmeans.cluster_centers_[cluster_idx]
                top_words_idx = cluster_center.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                themes[f'Theme_{cluster_idx + 1}'] = {
                    'keywords': top_words,
                    'weight': cluster_center.max()
                }
            
            self.df['primary_theme'] = [f'Theme_{label + 1}' for label in cluster_labels]
            
            # Calculate confidence as distance to cluster center
            distances = kmeans.transform(tfidf_matrix)
            self.df['theme_confidence'] = [1 / (1 + np.min(dist)) for dist in distances]
        
        self.themes = themes
        
        # Theme distribution
        theme_counts = self.df['primary_theme'].value_counts()
        self.insights['theme_distribution'] = theme_counts.to_dict()
        
        for theme_name, theme_data in themes.items():
            self.logger.info(f"{theme_name}: {', '.join(theme_data['keywords'][:5])}")
    
    def identify_feature_requests(self) -> None:
        """Identify potential feature requests."""
        self.logger.info("Identifying feature requests...")
        
        feature_scores = []
        for text in self.df['cleaned_text']:
            score = sum(1 for keyword in self.feature_keywords if keyword in text.lower())
            feature_scores.append(score)
        
        self.df['feature_request_score'] = feature_scores
        self.df['is_feature_request'] = self.df['feature_request_score'] > 0
        
        feature_requests = self.df[self.df['is_feature_request']]
        self.insights['feature_requests'] = {
            'count': len(feature_requests),
            'percentage': len(feature_requests) / len(self.df) * 100,
            'top_requests': feature_requests.nlargest(5, 'feature_request_score')[self.text_column].tolist()
        }
        
        self.logger.info(f"Found {len(feature_requests)} potential feature requests "
                        f"({len(feature_requests) / len(self.df) * 100:.1f}%)")
    
    def identify_pain_points(self) -> None:
        """Identify pain points and issues."""
        self.logger.info("Identifying pain points...")
        
        pain_scores = []
        for text in self.df['cleaned_text']:
            score = sum(1 for keyword in self.pain_keywords if keyword in text.lower())
            pain_scores.append(score)
        
        self.df['pain_point_score'] = pain_scores
        self.df['is_pain_point'] = self.df['pain_point_score'] > 0
        
        pain_points = self.df[self.df['is_pain_point']]
        self.insights['pain_points'] = {
            'count': len(pain_points),
            'percentage': len(pain_points) / len(self.df) * 100,
            'top_pain_points': pain_points.nlargest(5, 'pain_point_score')[self.text_column].tolist()
        }
        
        self.logger.info(f"Found {len(pain_points)} potential pain points "
                        f"({len(pain_points) / len(self.df) * 100:.1f}%)")
    
    def run_analysis(self, num_themes: int = 5, theme_method: str = 'lda') -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        self.load_data()
        self.preprocess_texts()
        self.analyze_sentiment()
        self.extract_themes(num_themes, theme_method)
        self.identify_feature_requests()
        self.identify_pain_points()
        
        # Generate summary insights
        self.insights['summary'] = {
            'total_feedback': len(self.df),
            'themes': self.themes,
            'avg_sentiment': self.df['polarity'].mean(),
            'most_common_theme': self.df['primary_theme'].mode()[0] if not self.df.empty else None
        }
        
        return self.insights


class FeedbackProcessor:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_csv(csv_path: str, text_column: str = None) -> Tuple[pd.DataFrame, str]:
        """Load and validate CSV file."""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {e}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Auto-detect text column if not specified
        if text_column is None:
            possible_columns = ['feedback', 'text', 'comment', 'review', 'description']
            text_column = None
            for col in possible_columns:
                if col.lower() in [c.lower() for c in df.columns]:
                    text_column = col
                    break
            
            if text_column is None:
                raise ValueError(f"Could not find text column. Available columns: {list(df.columns)}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        # Remove empty rows
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].astype(str).str.strip() != '']
        
        if df.empty:
            raise ValueError("No valid feedback text found in CSV")
        
        return df, text_column


class TextProcessor:
    """Handles text preprocessing and cleaning."""
    
    def __init__(self, lemmatizer, stop_words):
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        
        # Tokenize
        tokens = word_tokenize(cleaned)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)


class ReportGenerator:
    """Generates detailed reports and visualizations."""
    
    def __init__(self, analyzer: FeedbackAnalyzer, output_dir: str = 'feedback_analysis_output'):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive text summary report."""
        insights = self.analyzer.insights
        df = self.analyzer.df
        
        report = []
        report.append("# User Feedback Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Source file: {self.analyzer.csv_path}")
        report.append("")
        
        # Overview
        report.append("## Overview")
        report.append(f"- Total feedback entries: {len(df)}")
        report.append(f"- Average sentiment score: {insights['summary']['avg_sentiment']:.3f}")
        report.append(f"- Most common theme: {insights['summary']['most_common_theme']}")
        report.append("")
        
        # Sentiment Analysis
        report.append("## Sentiment Distribution")
        for sentiment, count in insights['sentiment_distribution'].items():
            percentage = count / len(df) * 100
            report.append(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Themes
        report.append("## Identified Themes")
        for theme_name, theme_data in self.analyzer.themes.items():
            report.append(f"### {theme_name}")
            report.append(f"Keywords: {', '.join(theme_data['keywords'][:8])}")
            theme_count = insights['theme_distribution'].get(theme_name, 0)
            percentage = theme_count / len(df) * 100
            report.append(f"Frequency: {theme_count} feedback entries ({percentage:.1f}%)")
            report.append("")
        
        # Feature Requests
        report.append("## Feature Requests")
        fr_data = insights['feature_requests']
        report.append(f"- Total feature requests: {fr_data['count']} ({fr_data['percentage']:.1f}%)")
        report.append("### Top Feature Requests:")
        for i, request in enumerate(fr_data['top_requests'][:5], 1):
            report.append(f"{i}. {request[:200]}{'...' if len(request) > 200 else ''}")
        report.append("")
        
        # Pain Points
        report.append("## Pain Points")
        pp_data = insights['pain_points']
        report.append(f"- Total pain points: {pp_data['count']} ({pp_data['percentage']:.1f}%)")
        report.append("### Top Pain Points:")
        for i, pain_point in enumerate(pp_data['top_pain_points'][:5], 1):
            report.append(f"{i}. {pain_point[:200]}{'...' if len(pain_point) > 200 else ''}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        # Sentiment-based recommendations
        neg_pct = insights['sentiment_distribution'].get('negative', 0) / len(df) * 100
        if neg_pct > 30:
            report.append("- **High Priority**: Address negative sentiment (>30% of feedback)")
        
        # Feature request recommendations
        if fr_data['percentage'] > 20:
            report.append("- **Feature Development**: Strong demand for new features detected")
        
        # Pain point recommendations
        if pp_data['percentage'] > 15:
            report.append("- **Issue Resolution**: Significant pain points need attention")
        
        # Theme-based recommendations
        sorted_themes = sorted(insights['theme_distribution'].items(), key=lambda x: x[1], reverse=True)
        if sorted_themes:
            top_theme = sorted_themes[0][0]
            report.append(f"- **Focus Area**: Prioritize '{top_theme}' (most common theme)")
        
        return '\n'.join(report)
    
    def save_detailed_data(self) -> None:
        """Save detailed analysis data to CSV."""
        output_file = self.output_dir / 'detailed_analysis.csv'
        self.analyzer.df.to_csv(output_file, index=False)
        print(f"Detailed data saved to: {output_file}")
    
    def save_insights_json(self) -> None:
        """Save insights as JSON."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        
        insights_copy = {}
        for key, value in self.analyzer.insights.items():
            if isinstance(value, dict):
                insights_copy[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                insights_copy[key] = convert_numpy(value)
        
        output_file = self.output_dir / 'insights.json'
        with open(output_file, 'w') as f:
            json.dump(insights_copy, f, indent=2, default=str)
        print(f"Insights saved to: {output_file}")
    
    def create_visualizations(self) -> None:
        """Create visualization plots."""
        if plt is None:
            print("Matplotlib not available. Skipping visualizations.")
            return
        
        df = self.analyzer.df
        insights = self.analyzer.insights
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('User Feedback Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution
        sentiment_data = insights['sentiment_distribution']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        axes[0, 0].pie(sentiment_data.values(), labels=sentiment_data.keys(), autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Theme Distribution
        theme_data = insights['theme_distribution']
        theme_names = list(theme_data.keys())
        theme_counts = list(theme_data.values())
        
        bars = axes[0, 1].bar(range(len(theme_names)), theme_counts, color='#96ceb4')
        axes[0, 1].set_xlabel('Themes')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Theme Distribution')
        axes[0, 1].set_xticks(range(len(theme_names)))
        axes[0, 1].set_xticklabels(theme_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom')
        
        # 3. Sentiment vs Theme
        sentiment_theme = df.groupby(['primary_theme', 'sentiment']).size().unstack(fill_value=0)
        sentiment_theme.plot(kind='bar', ax=axes[1, 0], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        axes[1, 0].set_title('Sentiment by Theme')
        axes[1, 0].set_xlabel('Theme')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Sentiment')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Feature Requests vs Pain Points
        categories = ['Feature Requests', 'Pain Points', 'Other']
        counts = [
            insights['feature_requests']['count'],
            insights['pain_points']['count'],
            len(df) - insights['feature_requests']['count'] - insights['pain_points']['count']
        ]
        
        bars = axes[1, 1].bar(categories, counts, color=['#f7b731', '#ff6b6b', '#95a5a6'])
        axes[1, 1].set_title('Feature Requests vs Pain Points')
        axes[1, 1].set_ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'feedback_analysis_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_file}")
        
        # Create word cloud for themes if possible
        try:
            from wordcloud import WordCloud
            
            # Combine all theme keywords
            all_keywords = []
            for theme_data in self.analyzer.themes.values():
                all_keywords.extend(theme_data['keywords'])
            
            if all_keywords:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_keywords))
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Theme Keywords Word Cloud', fontsize=16, fontweight='bold')
                
                output_file = self.output_dir / 'theme_wordcloud.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Word cloud saved to: {output_file}")
                
        except ImportError:
            print("WordCloud not available. Install with: pip install wordcloud")


def main():
    """Main function to run the feedback analyzer."""
    parser = argparse.ArgumentParser(description='Analyze user feedback from CSV files')
    parser.add_argument('csv_file', help='Path to CSV file containing feedback')
    parser.add_argument('--text-column', help='Name of column containing feedback text')
    parser.add_argument('--output-dir', default='feedback_analysis_output', 
                       help='Directory to save analysis results')
    parser.add_argument('--num-themes', type=int, default=5, 
                       help='Number of themes to extract')
    parser.add_argument('--theme-method', choices=['lda', 'kmeans'], default='lda',
                       help='Method for theme extraction (lda or kmeans)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualization plots')
    parser.add_argument('--save-data', action='store_true',
                       help='Save detailed analysis data to CSV')
    
    args = parser.parse_args()
    
    try:
        print(f"🔍 Starting feedback analysis...")
        print(f"📁 Input file: {args.csv_file}")
        print(f"📊 Output directory: {args.output_dir}")
        print()
        
        # Initialize analyzer
        analyzer = FeedbackAnalyzer(args.csv_file, args.text_column)
        
        # Run complete analysis
        insights = analyzer.run_analysis(args.num_themes, args.theme_method)
        
        # Generate reports
        report_generator = ReportGenerator(analyzer, args.output_dir)
        
        # Generate and save summary report
        summary_report = report_generator.generate_summary_report()
        report_file = Path(args.output_dir) / 'feedback_analysis_report.md'
        with open(report_file, 'w') as f:
            f.write(summary_report)
        print(f"📋 Summary report saved to: {report_file}")
        
        # Save insights as JSON
        report_generator.save_insights_json()
        
        # Save detailed data if requested
        if args.save_data:
            report_generator.save_detailed_data()
        
        # Create visualizations if requested
        if args.visualize:
            report_generator.create_visualizations()
        
        # Print key insights to console
        print("\n" + "="*60)
        print("🎯 KEY INSIGHTS")
        print("="*60)
        
        summary = insights['summary']
        print(f"📈 Total feedback entries: {summary['total_feedback']}")
        print(f"😊 Average sentiment: {summary['avg_sentiment']:.3f}")
        print(f"🏷️  Most common theme: {summary['most_common_theme']}")
        
        print(f"\n💡 Feature requests: {insights['feature_requests']['count']} "
              f"({insights['feature_requests']['percentage']:.1f}%)")
        print(f"⚠️  Pain points: {insights['pain_points']['count']} "
              f"({insights['pain_points']['percentage']:.1f}%)")
        
        print(f"\n📊 Sentiment breakdown:")
        for sentiment, count in insights['sentiment_distribution'].items():
            percentage = count / summary['total_feedback'] * 100
            print(f"   {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 Top themes:")
        sorted_themes = sorted(insights['theme_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)
        for i, (theme, count) in enumerate(sorted_themes[:3], 1):
            percentage = count / summary['total_feedback'] * 100
            keywords = analyzer.themes[theme]['keywords'][:3]
            print(f"   {i}. {theme}: {count} ({percentage:.1f}%) - {', '.join(keywords)}")
        
        print(f"\n✅ Analysis complete! Check '{args.output_dir}' for detailed results.")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{args.csv_file}' not found.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())