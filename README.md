# User Feedback Analysis Tool

A comprehensive Python script that processes user feedback CSV files to extract themes, sentiment, feature requests, and pain points using Natural Language Processing techniques.

## Features

- **Automated Theme Extraction**: Uses LDA (Latent Dirichlet Allocation) or K-means clustering to identify common themes
- **Sentiment Analysis**: Classifies feedback as positive, negative, or neutral
- **Feature Request Detection**: Identifies potential feature requests using keyword matching
- **Pain Point Analysis**: Detects issues and problems mentioned in feedback
- **Comprehensive Reporting**: Generates detailed reports in Markdown, JSON, and CSV formats
- **Visualizations**: Creates charts and word clouds to visualize insights
- **Flexible Input**: Auto-detects text columns or allows manual specification

## Installation

1. Clone or download the script files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data (done automatically on first run):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Usage

### Basic Usage

```bash
python feedback_analyzer.py sample_feedback.csv
```

### Advanced Usage

```bash
python feedback_analyzer.py feedback.csv \
    --text-column "feedback" \
    --output-dir "analysis_results" \
    --num-themes 8 \
    --theme-method lda \
    --visualize \
    --save-data
```

### Command Line Arguments

- `csv_file`: Path to the CSV file containing feedback (required)
- `--text-column`: Name of the column containing feedback text (auto-detected if not specified)
- `--output-dir`: Directory to save analysis results (default: "feedback_analysis_output")
- `--num-themes`: Number of themes to extract (default: 5)
- `--theme-method`: Method for theme extraction - "lda" or "kmeans" (default: "lda")
- `--visualize`: Generate visualization plots
- `--save-data`: Save detailed analysis data to CSV

## CSV File Format

The CSV file should contain user feedback with the following requirements:

### Required Columns
- A text column containing feedback (common names: 'feedback', 'text', 'comment', 'review', 'description')

### Optional Columns
- `user_id`: Unique identifier for users
- `timestamp`: When the feedback was submitted
- `rating`: Numerical rating (1-5, 1-10, etc.)
- `category`: Pre-existing categories

### Example CSV Structure

```csv
feedback,user_id,timestamp,rating
"Love this app! Very intuitive.",user_001,2024-01-15 10:30:00,5
"App crashes when uploading files.",user_002,2024-01-15 11:45:00,1
"Need dark mode feature please.",user_003,2024-01-15 14:20:00,3
```

## Output Files

The tool generates several output files in the specified output directory:

### 1. Summary Report (`feedback_analysis_report.md`)
- Comprehensive analysis summary
- Theme descriptions with keywords
- Feature request and pain point summaries
- Actionable recommendations

### 2. Insights Data (`insights.json`)
- Structured data in JSON format
- Sentiment distribution
- Theme statistics
- Feature request and pain point counts

### 3. Detailed Analysis (`detailed_analysis.csv`) - Optional
- Original data with added analysis columns
- Sentiment scores and classifications
- Theme assignments
- Feature request and pain point flags

### 4. Visualizations - Optional
- `feedback_analysis_dashboard.png`: Multi-panel dashboard
- `theme_wordcloud.png`: Word cloud of theme keywords

## Analysis Components

### Theme Extraction

The tool uses two methods for theme extraction:

1. **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling
2. **K-means Clustering**: Document clustering based on TF-IDF vectors

### Sentiment Analysis

Uses TextBlob for sentiment polarity analysis:
- **Positive**: Polarity > 0.1
- **Negative**: Polarity < -0.1
- **Neutral**: -0.1 ≤ Polarity ≤ 0.1

### Feature Request Detection

Identifies feedback containing keywords such as:
- feature, add, would like, wish, want, need
- enhancement, improvement, suggest, request
- integrate, implement, support, include

### Pain Point Detection

Detects issues using keywords like:
- bug, error, issue, problem, difficult
- slow, confusing, frustrating, broken
- crash, freeze, hang, terrible

## Example Output

```
🎯 KEY INSIGHTS
============================================================
📈 Total feedback entries: 20
😊 Average sentiment: 0.142
🏷️  Most common theme: Theme_1

💡 Feature requests: 6 (30.0%)
⚠️  Pain points: 8 (40.0%)

📊 Sentiment breakdown:
   Positive: 8 (40.0%)
   Negative: 6 (30.0%)
   Neutral: 6 (30.0%)

🎯 Top themes:
   1. Theme_1: 7 (35.0%) - app, feature, good
   2. Theme_2: 5 (25.0%) - problem, issue, bug
   3. Theme_3: 4 (20.0%) - interface, design, user
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **nltk**: Natural language processing
- **scikit-learn**: Machine learning algorithms
- **textblob**: Text processing and sentiment analysis
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **wordcloud**: Word cloud generation (optional)

## Troubleshooting

### Common Issues

1. **"Column not found" error**: Specify the correct text column name with `--text-column`
2. **NLTK data not found**: Run the script once to auto-download required data
3. **Memory issues with large files**: Process data in chunks or reduce the number of features
4. **Visualization errors**: Install matplotlib and seaborn, or run without `--visualize`

### Performance Tips

- For large datasets (>10,000 entries), consider reducing `--num-themes`
- Use K-means clustering for faster processing on very large datasets
- Process data in batches if memory becomes an issue

## Customization

The script can be easily customized by modifying:

- **Feature request keywords**: Edit `feature_keywords` list in `FeedbackAnalyzer.__init__()`
- **Pain point keywords**: Edit `pain_keywords` list in `FeedbackAnalyzer.__init__()`
- **Sentiment thresholds**: Modify polarity thresholds in `analyze_sentiment()`
- **Text preprocessing**: Customize cleaning rules in `TextProcessor.clean_text()`

## License

This tool is provided as-is for educational and commercial use. Feel free to modify and distribute.

## Contributing

To contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example files provided
3. Ensure all dependencies are correctly installed