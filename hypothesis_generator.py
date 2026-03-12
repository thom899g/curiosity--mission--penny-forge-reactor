"""
Hypothesis Generator - The "Nucleus" of the Curiosity Reactor.
Consumes unstructured data to generate testable scientific hypotheses.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from firebase_config import get_firestore

# Download NLTK data (with error handling)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    """Data class representing a scientific hypothesis."""
    hypothesis_id: str
    statement: str
    confidence_prior: float  # Prior probability (0-1)
    test_plan: str  # Which experiment runner to use
    budget_requested: float
    created_at: datetime
    status: str  # pending_funding, funded, testing, validated, invalidated
    data_source: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-compatible dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

class DataIngestor:
    """Ingests data from various free sources."""
    
    FREE_DATA_SOURCES = {
        'rss_financial': [
            'https://rss.cnn.com/rss/money_news_international.rss',
            'https://feeds.bbci.co.uk/news/business/rss.xml'
        ],
        'crypto_sentiment': [
            'https://coinmarketcap.com/headlines/news/',
            'https://cryptopanic.com/news/'
        ],
        'public_datasets': [
            'https://datahub.io/core/economic-indicators/r/1.csv',
            'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        ]
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CuriosityReactor/1.0 (Scientific Research)'
        })
        
    def fetch_rss_feed(self, url: str) -> List[Dict[str, str]]:
        """Fetch and parse RSS feed content."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = []
            
            for item in soup.find_all('item')[:10]:  # Limit to 10 items
                title = item.find('title')
                description = item.find('description')
                pub_date = item.find('pubDate')
                
                items.append({
                    'title': title.text if title else '',
                    'description': description.text if description else '',
                    'pub_date': pub_date.text if pub_date else '',
                    'source': url,
                    'fetched_at': datetime.utcnow().isoformat()
                })
            
            logger.info(f"Fetched {len(items)} items from {url}")
            return items
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch RSS feed {url}: {str(e)}")
            return []
    
    def scrape_crypto_news(self) -> List[Dict[str, Any]]:
        """Scrape cryptocurrency news headlines."""
        # Using Cryptopanic public API (free tier)
        try:
            response = self.session.get(
                'https://cryptopanic.com/api/v1/posts/',
                params={'auth_token': 'public', 'kind': 'news'},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            for post in data.get('results', [])[:15]:
                news_items.append({
                    'title': post.get('title', ''),
                    'published_at': post.get('published_at', ''),
                    'source': 'cryptopanic',
                    'sentiment': post.get('sentiment', 'neutral'),
                    'votes': post.get('votes', {})
                })
            
            logger.info(f"Fetched {len(news_items)} crypto news items")
            return news_items
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch crypto news: {str(e)}")
            return []
    
    def ingest_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Ingest data from all sources."""
        all_data = {}
        
        # Fetch RSS feeds
        financial_news = []
        for url in self.FREE_DATA_SOURCES['rss_financial']:
            financial_news.extend(self.fetch_rss_feed(url))
        all_data['financial_news'] = financial_news
        
        # Fetch crypto news
        crypto_news = self.scrape_crypto_news()
        all_data['crypto_news'] = crypto_news
        
        logger.info(f"Total ingested: {sum(len(v) for v in all_data.values())} items")
        return all_data

class HypothesisEngine:
    """Generates testable hypotheses from ingested data."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_sentiment_correlation(self, news_items: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Generate hypotheses about sentiment correlations."""
        hypotheses = []
        
        if not news_items or len(news_items) < 10:
            return hypotheses
        
        # Extract sentiment scores
        sentiments = []
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('description', '')}"
            score = self.sia.polarity_scores(text)
            sentiments.append(score['compound'])
        
        # Look for patterns
        if len(sentiments) >= 20:
            # Simple pattern detection: consecutive similar sentiments
            window_size = 5
            for i in range(len(sentiments) - window_size):
                window = sentiments[i:i+window_size]
                if all(s > 0.5 for s in window):  # Strong positive cluster
                    hyp = Hypothesis(
                        hypothesis_id=f"HYP_SENT_POS_{uuid.uuid4().hex[:8]}",
                        statement=f"Cluster of {window_size} extremely positive news items predicts market uptick in next 24 hours",
                        confidence_prior=0.4,
                        test_plan="sentiment_price_correlation",
                        budget_requested=0.50,
                        created_at=datetime.utcnow(),
                        status="pending_funding",
                        data_source="financial_news",
                        metadata={
                            'sentiment_window': window,
                            'pattern_type': 'positive_cluster',
                            'window_size': window_size
                        }
                    )
                    hypotheses.append(hyp)
                    break
        
        return hypotheses
    
    def detect_anomalous_patterns(self, crypto_news: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Detect anomalous patterns in crypto news."""
        hypotheses = []
        
        if not crypto_news or len(crypto_news) < 15:
            return hypotheses
        
        try:
            # Create feature matrix from news titles
            titles = [item.get('title', '') for item in crypto_news]
            if len(titles) < 10:
                return hypotheses
                
            # Simple feature: title length and word count
            features = []
            for title in titles:
                features.append([
                    len(title),
                    len(title.split()),
                    1 if '!' in title else 0,
                    1 if '?' in title else 0
                ])
            
            features = np.array(features)
            
            # Detect anomalies
            if len(features) > 10:
                labels = self.anomaly_detector.fit_predict(features)
                anomaly_indices = np.where(labels == -1)[0]
                
                for idx in anomaly_indices[:3]:  # Limit to top 3 anomalies
                    if idx < len(crypto_news):
                        news = crypto_news[idx]
                        hyp = Hypothesis(
                            hypothesis_id=f"HYP_ANOM_{uuid.uuid4().hex[:8]}",
                            statement=f"Anomalous news pattern '{news.get('title', '')[:50]}...' predicts unusual market movement",
                            confidence_prior=0.3,
                            test_plan="anomaly_market_correlation",
                            budget_requested=0.50,
                            created_at=datetime.utcnow(),
                            status="pending_funding",
                            data_source="crypto_news",
                            metadata={
                                'anomaly_score': float(self.anomaly_detector.score_samples([features[idx]])[0]),
                                'title': news.get('title', ''),
                                'sentiment': news.get('sentiment', 'neutral')
                            }
                        )
                        hypotheses.append(hyp)
                        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
        
        return hypotheses
    
    def generate_from_data_correlation(self) -> List[Hypothesis]:
        """Generate hypotheses from public dataset correlations."""
        hypotheses = []
        
        # Example: Hypothesize about economic indicators
        # In production, this would fetch actual datasets
        example_hypotheses = [
            Hypothesis(
                hypothesis_id=f"HYP_ECON_{uuid.uuid4().hex[:8]}