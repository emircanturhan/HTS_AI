import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import feedparser
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cloudscraper
from newspaper import Article, Config as NewspaperConfig
import trafilatura
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import time
import hashlib
from urllib.parse import urlparse, urljoin
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
from fake_useragent import UserAgent
import robotparser
from ratelimit import limits, sleep_and_retry
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class NewsSource:
    """Haber kaynağı yapısı"""
    name: str
    source_type: str  # 'api', 'rss', 'web_scrape', 'social'
    url: str
    api_key: Optional[str] = None
    selectors: Optional[Dict] = None  # For web scraping
    rate_limit: int = 60  # Requests per minute
    priority: int = 1  # 1-5, higher is more important
    categories: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ['en'])
    active: bool = True
    last_fetched: Optional[datetime] = None
    error_count: int = 0
    
@dataclass
class FetchedArticle:
    """Çekilen makale yapısı"""
    title: str
    content: str
    url: str
    source: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[float] = None
    language: str = 'en'
    raw_html: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    hash_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.hash_id:
            # Generate unique hash for deduplication
            content = f"{self.title}{self.url}{self.published_date}"
            self.hash_id = hashlib.sha256(content.encode()).hexdigest()[:16]

# ============================================
# ENHANCED DATA FETCHER
# ============================================

class EnhancedDataFetcher:
    """Gelişmiş çok kaynaklı veri çekme sistemi"""
    
    def __init__(self):
        self.sources = self._initialize_sources()
        self.ua = UserAgent()
        self.session = None
        self.scraper = cloudscraper.create_scraper()
        self.article_cache = {}
        self.rate_limiters = {}
        self.robots_checkers = {}
        
        # Selenium setup for dynamic content
        self.selenium_driver = None
        self._setup_selenium()
        
        # Thread pool for parallel fetching
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Statistics
        self.stats = defaultdict(lambda: {
            'total_fetched': 0,
            'errors': 0,
            'last_success': None
        })
    
    def _initialize_sources(self) -> Dict[str, NewsSource]:
        """Tüm haber kaynaklarını başlat"""
        
        sources = {}
        
        # ========== API SOURCES ==========
        sources['coindesk_api'] = NewsSource(
            name='CoinDesk API',
            source_type='api',
            url='https://api.coindesk.com/v1/news',
            categories=['crypto', 'bitcoin', 'ethereum'],
            priority=5
        )
        
        sources['cryptopanic_api'] = NewsSource(
            name='CryptoPanic API',
            source_type='api',
            url='https://cryptopanic.com/api/v1/posts/',
            api_key='YOUR_API_KEY',
            categories=['crypto', 'news', 'social'],
            priority=4
        )
        
        sources['newsapi'] = NewsSource(
            name='NewsAPI',
            source_type='api',
            url='https://newsapi.org/v2/everything',
            api_key='YOUR_API_KEY',
            categories=['crypto', 'finance', 'technology'],
            languages=['en', 'tr'],
            priority=4
        )
        
        sources['alphavantage'] = NewsSource(
            name='Alpha Vantage',
            source_type='api',
            url='https://www.alphavantage.co/query',
            api_key='YOUR_API_KEY',
            categories=['market', 'forex', 'crypto'],
            priority=3
        )
        
        sources['messari'] = NewsSource(
            name='Messari API',
            source_type='api',
            url='https://data.messari.io/api/v1/news',
            categories=['crypto', 'research', 'analysis'],
            priority=5
        )
        
        # ========== RSS FEEDS ==========
        sources['coindesk_rss'] = NewsSource(
            name='CoinDesk RSS',
            source_type='rss',
            url='https://www.coindesk.com/arc/outboundfeeds/rss/',
            categories=['crypto', 'news'],
            priority=5
        )
        
        sources['cointelegraph_rss'] = NewsSource(
            name='Cointelegraph RSS',
            source_type='rss',
            url='https://cointelegraph.com/rss',
            categories=['crypto', 'news', 'analysis'],
            priority=4
        )
        
        sources['bitcoin_magazine_rss'] = NewsSource(
            name='Bitcoin Magazine RSS',
            source_type='rss',
            url='https://bitcoinmagazine.com/feed',
            categories=['bitcoin', 'analysis'],
            priority=4
        )
        
        sources['decrypt_rss'] = NewsSource(
            name='Decrypt RSS',
            source_type='rss',
            url='https://decrypt.co/feed',
            categories=['crypto', 'defi', 'nft'],
            priority=3
        )
        
        sources['theblock_rss'] = NewsSource(
            name='The Block RSS',
            source_type='rss',
            url='https://www.theblock.co/rss.xml',
            categories=['crypto', 'research', 'data'],
            priority=5
        )
        
        sources['bloomberg_crypto_rss'] = NewsSource(
            name='Bloomberg Crypto RSS',
            source_type='rss',
            url='https://www.bloomberg.com/crypto/rss',
            categories=['crypto', 'finance', 'markets'],
            priority=5
        )
        
        # Türkçe RSS Kaynakları
        sources['btchaber_rss'] = NewsSource(
            name='BTCHaber RSS',
            source_type='rss',
            url='https://www.btchaber.com/feed/',
            categories=['crypto', 'turkish'],
            languages=['tr'],
            priority=3
        )
        
        sources['koinbulteni_rss'] = NewsSource(
            name='Koin Bülteni RSS',
            source_type='rss',
            url='https://koinbulteni.com/feed',
            categories=['crypto', 'turkish'],
            languages=['tr'],
            priority=3
        )
        
        # ========== WEB SCRAPING SOURCES ==========
        sources['reuters_crypto'] = NewsSource(
            name='Reuters Crypto',
            source_type='web_scrape',
            url='https://www.reuters.com/technology/crypto',
            selectors={
                'articles': 'article.story',
                'title': 'h3.story-title',
                'content': 'div.story-body',
                'date': 'time.timestamp',
                'author': 'span.author-name'
            },
            categories=['crypto', 'finance', 'regulation'],
            priority=5
        )
        
        sources['wsj_crypto'] = NewsSource(
            name='WSJ Cryptocurrencies',
            source_type='web_scrape',
            url='https://www.wsj.com/news/types/cryptocurrencies',
            selectors={
                'articles': 'article.WSJTheme--story',
                'title': 'h3.WSJTheme--headline',
                'content': 'div.WSJTheme--body',
                'date': 'time.WSJTheme--timestamp'
            },
            categories=['crypto', 'finance', 'institutional'],
            priority=5
        )
        
        sources['forbes_crypto'] = NewsSource(
            name='Forbes Crypto',
            source_type='web_scrape',
            url='https://www.forbes.com/crypto-blockchain/',
            selectors={
                'articles': 'article.stream-item',
                'title': 'h4.stream-item__title',
                'content': 'div.stream-item__description',
                'date': 'time.stream-item__date',
                'author': 'span.stream-author__name'
            },
            categories=['crypto', 'business', 'innovation'],
            priority=4
        )
        
        sources['coinmarketcap_news'] = NewsSource(
            name='CoinMarketCap News',
            source_type='web_scrape',
            url='https://coinmarketcap.com/headlines/news/',
            selectors={
                'articles': 'div.news-list-item',
                'title': 'h3.news-title',
                'content': 'div.news-content',
                'date': 'span.news-date'
            },
            categories=['crypto', 'market', 'data'],
            priority=3
        )
        
        sources['defipulse_news'] = NewsSource(
            name='DeFi Pulse News',
            source_type='web_scrape',
            url='https://defipulse.com/blog',
            selectors={
                'articles': 'article.post',
                'title': 'h2.post-title',
                'content': 'div.post-content',
                'date': 'time.post-date'
            },
            categories=['defi', 'protocols', 'tvl'],
            priority=4
        )
        
        # Reddit (etik kullanım)
        sources['reddit_cryptocurrency'] = NewsSource(
            name='Reddit r/cryptocurrency',
            source_type='web_scrape',
            url='https://old.reddit.com/r/cryptocurrency/top/',
            selectors={
                'articles': 'div.thing',
                'title': 'a.title',
                'content': 'div.usertext-body',
                'date': 'time.live-timestamp',
                'author': 'a.author'
            },
            categories=['crypto', 'social', 'sentiment'],
            rate_limit=30,  # Be respectful
            priority=2
        )
        
        return sources
    
    def _setup_selenium(self):
        """Selenium WebDriver kurulumu"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.ua.random}')
            
            # Disable images for faster loading
            prefs = {"profile.managed_default_content_settings.images": 2}
            options.add_experimental_option("prefs", prefs)
            
            self.selenium_driver = webdriver.Chrome(options=options)
            logger.info("Selenium WebDriver initialized")
        except Exception as e:
            logger.warning(f"Selenium setup failed: {e}. Dynamic content scraping disabled.")
    
    async def fetch_all(self, categories: Optional[List[str]] = None,
                       priority_min: int = 1) -> List[FetchedArticle]:
        """Tüm kaynaklardan veri çek"""
        
        articles = []
        
        # Filter sources by category and priority
        active_sources = [
            source for source in self.sources.values()
            if source.active and 
               source.priority >= priority_min and
               (not categories or any(cat in source.categories for cat in categories))
        ]
        
        # Create async session
        async with ClientSession(timeout=ClientTimeout(total=30)) as self.session:
            # Fetch from all sources concurrently
            tasks = []
            
            for source in active_sources:
                if source.source_type == 'api':
                    tasks.append(self._fetch_from_api(source))
                elif source.source_type == 'rss':
                    tasks.append(self._fetch_from_rss(source))
                elif source.source_type == 'web_scrape':
                    tasks.append(self._fetch_from_web(source))
            
            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Fetch error: {result}")
        
        # Deduplicate articles
        unique_articles = self._deduplicate_articles(articles)
        
        logger.info(f"Fetched {len(unique_articles)} unique articles from {len(active_sources)} sources")
        
        return unique_articles
    
    async def _fetch_from_api(self, source: NewsSource) -> List[FetchedArticle]:
        """API'den veri çek"""
        
        articles = []
        
        try:
            # Check rate limit
            if not self._check_rate_limit(source):
                logger.warning(f"Rate limit reached for {source.name}")
                return articles
            
            # Prepare headers
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'application/json'
            }
            
            # Add API key if required
            params = {}
            if source.api_key:
                if source.name == 'NewsAPI':
                    headers['X-Api-Key'] = source.api_key
                else:
                    params['apikey'] = source.api_key
            
            # Make request
            async with self.session.get(source.url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse based on source
                    if source.name == 'CoinDesk API':
                        articles = self._parse_coindesk_api(data)
                    elif source.name == 'CryptoPanic API':
                        articles = self._parse_cryptopanic_api(data)
                    elif source.name == 'NewsAPI':
                        articles = self._parse_newsapi(data)
                    elif source.name == 'Messari API':
                        articles = self._parse_messari_api(data)
                    else:
                        articles = self._parse_generic_api(data, source)
                    
                    # Update statistics
                    self.stats[source.name]['total_fetched'] += len(articles)
                    self.stats[source.name]['last_success'] = datetime.now()
                    source.last_fetched = datetime.now()
                else:
                    logger.error(f"API error for {source.name}: {response.status}")
                    source.error_count += 1
                    
        except Exception as e:
            logger.error(f"Error fetching from {source.name}: {e}")
            source.error_count += 1
            self.stats[source.name]['errors'] += 1
        
        return articles
    
    async def _fetch_from_rss(self, source: NewsSource) -> List[FetchedArticle]:
        """RSS feed'den veri çek"""
        
        articles = []
        
        try:
            # Check rate limit
            if not self._check_rate_limit(source):
                return articles
            
            # Fetch RSS feed
            headers = {'User-Agent': self.ua.random}
            
            async with self.session.get(source.url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse RSS
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:50]:  # Limit to 50 entries
                        article = await self._parse_rss_entry(entry, source)
                        if article:
                            articles.append(article)
                    
                    # Update stats
                    self.stats[source.name]['total_fetched'] += len(articles)
                    self.stats[source.name]['last_success'] = datetime.now()
                    source.last_fetched = datetime.now()
                else:
                    logger.error(f"RSS fetch error for {source.name}: {response.status}")
                    source.error_count += 1
                    
        except Exception as e:
            logger.error(f"Error fetching RSS from {source.name}: {e}")
            source.error_count += 1
            self.stats[source.name]['errors'] += 1
        
        return articles
    
    async def _fetch_from_web(self, source: NewsSource) -> List[FetchedArticle]:
        """Web scraping ile veri çek"""
        
        articles = []
        
        try:
            # Check robots.txt
            if not self._check_robots_txt(source.url):
                logger.warning(f"Robots.txt disallows scraping {source.url}")
                return articles
            
            # Check rate limit
            if not self._check_rate_limit(source):
                return articles
            
            # Determine scraping method
            if self._requires_javascript(source.url):
                articles = await self._scrape_with_selenium(source)
            else:
                articles = await self._scrape_with_beautifulsoup(source)
            
            # Update stats
            self.stats[source.name]['total_fetched'] += len(articles)
            self.stats[source.name]['last_success'] = datetime.now()
            source.last_fetched = datetime.now()
            
        except Exception as e:
            logger.error(f"Error scraping {source.name}: {e}")
            source.error_count += 1
            self.stats[source.name]['errors'] += 1
        
        return articles
    
    async def _scrape_with_beautifulsoup(self, source: NewsSource) -> List[FetchedArticle]:
        """BeautifulSoup ile scraping"""
        
        articles = []
        
        try:
            # Use cloudscraper for anti-bot protection
            response = self.scraper.get(
                source.url,
                headers={'User-Agent': self.ua.random}
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find articles using selectors
                article_elements = soup.select(source.selectors.get('articles', 'article'))
                
                for element in article_elements[:30]:  # Limit to 30 articles
                    article = self._extract_article_from_element(element, source, soup)
                    if article:
                        # Fetch full content if needed
                        if article.url and not article.content:
                            full_article = await self._fetch_full_article(article.url)
                            if full_article:
                                article.content = full_article.content
                                article.summary = full_article.summary
                        
                        articles.append(article)
                        
        except Exception as e:
            logger.error(f"BeautifulSoup scraping error: {e}")
        
        return articles
    
    async def _scrape_with_selenium(self, source: NewsSource) -> List[FetchedArticle]:
        """Selenium ile dinamik içerik scraping"""
        
        articles = []
        
        if not self.selenium_driver:
            return articles
        
        try:
            # Load page
            self.selenium_driver.get(source.url)
            
            # Wait for content to load
            wait = WebDriverWait(self.selenium_driver, 10)
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, source.selectors.get('articles', 'article'))
            ))
            
            # Scroll to load more content if needed
            self._scroll_page()
            
            # Get page source
            soup = BeautifulSoup(self.selenium_driver.page_source, 'html.parser')
            
            # Extract articles
            article_elements = soup.select(source.selectors.get('articles', 'article'))
            
            for element in article_elements[:30]:
                article = self._extract_article_from_element(element, source, soup)
                if article:
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Selenium scraping error: {e}")
        
        return articles
    
    def _extract_article_from_element(self, element, source: NewsSource, 
                                     soup: BeautifulSoup) -> Optional[FetchedArticle]:
        """HTML elementinden makale çıkar"""
        
        try:
            # Extract basic fields
            title_elem = element.select_one(source.selectors.get('title', 'h2'))
            title = title_elem.get_text(strip=True) if title_elem else None
            
            if not title:
                return None
            
            # URL
            url = None
            if title_elem and title_elem.name == 'a':
                url = urljoin(source.url, title_elem.get('href', ''))
            else:
                link_elem = element.select_one('a')
                if link_elem:
                    url = urljoin(source.url, link_elem.get('href', ''))
            
            # Content
            content_elem = element.select_one(source.selectors.get('content', 'p'))
            content = content_elem.get_text(strip=True) if content_elem else ''
            
            # Date
            date_elem = element.select_one(source.selectors.get('date', 'time'))
            published_date = self._parse_date(date_elem) if date_elem else None
            
            # Author
            author_elem = element.select_one(source.selectors.get('author', 'span.author'))
            author = author_elem.get_text(strip=True) if author_elem else None
            
            # Image
            image_elem = element.select_one('img')
            image_url = urljoin(source.url, image_elem.get('src', '')) if image_elem else None
            
            return FetchedArticle(
                title=title,
                content=content,
                url=url or source.url,
                source=source.name,
                author=author,
                published_date=published_date,
                categories=source.categories,
                image_url=image_url,
                language=source.languages[0] if source.languages else 'en',
                metadata={'source_type': source.source_type}
            )
            
        except Exception as e:
            logger.error(f"Error extracting article: {e}")
            return None
    
    async def _fetch_full_article(self, url: str) -> Optional[FetchedArticle]:
        """Makale içeriğinin tamamını çek"""
        
        try:
            # Try newspaper3k first
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            return FetchedArticle(
                title=article.title,
                content=article.text,
                url=url,
                source=urlparse(url).netloc,
                author=', '.join(article.authors) if article.authors else None,
                published_date=article.publish_date,
                tags=article.tags,
                image_url=article.top_image,
                summary=article.summary,
                metadata={'keywords': article.keywords}
            )
            
        except:
            # Fallback to trafilatura
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    content = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        deduplicate=True
                    )
                    
                    if content:
                        return FetchedArticle(
                            title='',
                            content=content,
                            url=url,
                            source=urlparse(url).netloc
                        )
            except:
                pass
        
        return None
    
    async def _parse_rss_entry(self, entry, source: NewsSource) -> Optional[FetchedArticle]:
        """RSS entry'yi parse et"""
        
        try:
            # Extract fields
            title = entry.get('title', '')
            url = entry.get('link', '')
            
            # Content extraction
            content = ''
            if 'content' in entry:
                content = entry.content[0].value if entry.content else ''
            elif 'summary' in entry:
                content = entry.summary
            elif 'description' in entry:
                content = entry.description
            
            # Clean HTML from content
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text(strip=True)
            
            # Date parsing
            published_date = None
            if 'published_parsed' in entry:
                published_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            elif 'updated_parsed' in entry:
                published_date = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
            
            # Author
            author = entry.get('author', None)
            
            # Categories/Tags
            tags = []
            if 'tags' in entry:
                tags = [tag.term for tag in entry.tags]
            
            # Media
            image_url = None
            if 'media_content' in entry:
                for media in entry.media_content:
                    if media.get('type', '').startswith('image'):
                        image_url = media.get('url')
                        break
            
            # Get full article content if needed
            if url and (not content or len(content) < 200):
                full_article = await self._fetch_full_article(url)
                if full_article and full_article.content:
                    content = full_article.content
            
            return FetchedArticle(
                title=title,
                content=content,
                url=url,
                source=source.name,
                author=author,
                published_date=published_date,
                categories=source.categories,
                tags=tags,
                image_url=image_url,
                language=source.languages[0] if source.languages else 'en',
                metadata={'feed_type': 'rss'}
            )
            
        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    def _parse_coindesk_api(self, data: Dict) -> List[FetchedArticle]:
        """CoinDesk API response parsing"""
        articles = []
        
        for item in data.get('data', []):
            article = FetchedArticle(
                title=item.get('title', ''),
                content=item.get('content', ''),
                url=item.get('url', ''),
                source='CoinDesk',
                author=item.get('author', {}).get('name'),
                published_date=datetime.fromisoformat(item.get('published_at', '')),
                categories=item.get('categories', []),
                tags=item.get('tags', []),
                image_url=item.get('thumbnail', '')
            )
            articles.append(article)
        
        return articles
    
    def _parse_cryptopanic_api(self, data: Dict) -> List[FetchedArticle]:
        """CryptoPanic API response parsing"""
        articles = []
        
        for item in data.get('results', []):
            article = FetchedArticle(
                title=item.get('title', ''),
                content=item.get('body', ''),
                url=item.get('url', ''),
                source=item.get('source', {}).get('title', 'CryptoPanic'),
                published_date=datetime.fromisoformat(item.get('published_at', '')),
                categories=['crypto'],
                metadata={
                    'votes': item.get('votes', {}),
                    'kind': item.get('kind', '')
                }
            )
            articles.append(article)
        
        return articles
    
    def _parse_newsapi(self, data: Dict) -> List[FetchedArticle]:
        """NewsAPI response parsing"""
        articles = []
        
        for item in data.get('articles', []):
            article = FetchedArticle(
                title=item.get('title', ''),
                content=item.get('content', item.get('description', '')),
                url=item.get('url', ''),
                source=item.get('source', {}).get('name', 'NewsAPI'),
                author=item.get('author'),
                published_date=datetime.fromisoformat(item.get('publishedAt', '')),
                image_url=item.get('urlToImage')
            )
            articles.append(article)
        
        return articles
    
    def _parse_messari_api(self, data: Dict) -> List[FetchedArticle]:
        """Messari API response parsing"""
        articles = []
        
        for item in data.get('data', []):
            article = FetchedArticle(
                title=item.get('title', ''),
                content=item.get('content', ''),
                url=item.get('url', ''),
                source='Messari',
                author=item.get('author', {}).get('name'),
                published_date=datetime.fromisoformat(item.get('published_at', '')),
                categories=item.get('tags', []),
                metadata={
                    'references': item.get('references', [])
                }
            )
            articles.append(article)
        
        return articles
    
    def _parse_generic_api(self, data: Dict, source: NewsSource) -> List[FetchedArticle]:
        """Generic API response parsing"""
        articles = []
        
        # Try to find the data array in common locations
        items = data.get('data', data.get('items', data.get('articles', data.get('results', []))))
        
        if isinstance(items, list):
            for item in items:
                article = FetchedArticle(
                    title=item.get('title', item.get('headline', '')),
                    content=item.get('content', item.get('body', item.get('description', ''))),
                    url=item.get('url', item.get('link', '')),
                    source=source.name,
                    author=item.get('author', item.get('creator')),
                    published_date=self._parse_date_string(
                        item.get('publishedAt', item.get('pubDate', item.get('created_at')))
                    ),
                    categories=source.categories
                )
                articles.append(article)
        
        return articles
    
    def _check_rate_limit(self, source: NewsSource) -> bool:
        """Rate limit kontrolü"""
        
        if source.name not in self.rate_limiters:
            self.rate_limiters[source.name] = {
                'requests': [],
                'limit': source.rate_limit
            }
        
        limiter = self.rate_limiters[source.name]
        now = datetime.now()
        
        # Remove requests older than 1 minute
        limiter['requests'] = [
            req for req in limiter['requests']
            if (now - req).total_seconds() < 60
        ]
        
        # Check if limit reached
        if len(limiter['requests']) >= limiter['limit']:
            return False
        
        # Add current request
        limiter['requests'].append(now)
        return True
    
    def _check_robots_txt(self, url: str) -> bool:
        """robots.txt kontrolü (etik scraping)"""
        
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        if robots_url not in self.robots_checkers:
            try:
                rp = robotparser.RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self.robots_checkers[robots_url] = rp
            except:
                # If robots.txt cannot be fetched, assume allowed
                return True
        
        checker = self.robots_checkers.get(robots_url)
        if checker:
            return checker.can_fetch(self.ua.random, url)
        
        return True
    
    def _requires_javascript(self, url: str) -> bool:
        """Sayfanın JavaScript gerektirip gerektirmediğini kontrol et"""
        
        # Sites known to require JavaScript
        js_required_domains = [
            'bloomberg.com',
            'wsj.com',
            'coinmarketcap.com',
            'twitter.com',
            'medium.com'
        ]
        
        parsed = urlparse(url)
        for domain in js_required_domains:
            if domain in parsed.netloc:
                return True
        
        return False
    
    def _scroll_page(self):
        """Selenium ile sayfayı kaydır (lazy loading için)"""
        
        if not self.selenium_driver:
            return
        
        try:
            # Scroll down multiple times
            last_height = self.selenium_driver.execute_script("return document.body.scrollHeight")
            
            for _ in range(3):  # Scroll 3 times
                # Scroll to bottom
                self.selenium_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait for content to load
                time.sleep(2)
                
                # Check if new content loaded
                new_height = self.selenium_driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                
        except Exception as e:
            logger.error(f"Scroll error: {e}")
    
    def _parse_date(self, element) -> Optional[datetime]:
        """HTML elementinden tarihi parse et"""
        
        if not element:
            return None
        
        try:
            # Try datetime attribute first
            if element.get('datetime'):
                return datetime.fromisoformat(element['datetime'].replace('Z', '+00:00'))
            
            # Try data attributes
            for attr in ['data-time', 'data-timestamp', 'data-date']:
                if element.get(attr):
                    # Could be Unix timestamp
                    try:
                        timestamp = int(element[attr])
                        if timestamp > 1000000000000:  # Milliseconds
                            timestamp = timestamp / 1000
                        return datetime.fromtimestamp(timestamp)
                    except:
                        pass
            
            # Try text content
            date_text = element.get_text(strip=True)
            return self._parse_date_string(date_text)
            
        except Exception as e:
            logger.debug(f"Date parsing error: {e}")
            return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Tarih string'ini parse et"""
        
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # Try relative dates (e.g., "2 hours ago")
        try:
            import dateparser
            return dateparser.parse(date_str)
        except:
            pass
        
        return None
    
    def _deduplicate_articles(self, articles: List[FetchedArticle]) -> List[FetchedArticle]:
        """Duplicate makaleleri temizle"""
        
        seen_hashes = set()
        seen_titles = set()
        unique = []
        
        for article in articles:
            # Check hash
            if article.hash_id in seen_hashes:
                continue
            
            # Check similar titles (fuzzy matching)
            title_lower = article.title.lower() if article.title else ''
            if any(self._similar_text(title_lower, seen) for seen in seen_titles):
                continue
            
            seen_hashes.add(article.hash_id)
            seen_titles.add(title_lower)
            unique.append(article)
        
        return unique
    
    def _similar_text(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """İki metnin benzerliğini kontrol et"""
        
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity > threshold

# ============================================
# SPECIALIZED FETCHERS
# ============================================

class SocialMediaFetcher:
    """Sosyal medya veri çekici"""
    
    def __init__(self):
        self.twitter_bearer = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    async def fetch_twitter_crypto(self, keywords: List[str] = None) -> List[FetchedArticle]:
        """Twitter'dan kripto tweetleri çek"""
        
        if not self.twitter_bearer:
            logger.warning("Twitter Bearer token not configured")
            return []
        
        articles = []
        
        if not keywords:
            keywords = ['bitcoin', 'ethereum', 'crypto', 'defi', 'nft']
        
        # Twitter API v2 endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        headers = {
            'Authorization': f'Bearer {self.twitter_bearer}'
        }
        
        for keyword in keywords:
            params = {
                'query': f'{keyword} -is:retweet lang:en',
                'max_results': 100,
                'tweet.fields': 'created_at,author_id,public_metrics'
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    for tweet in data.get('data', []):
                        article = FetchedArticle(
                            title=f"Tweet: {tweet['text'][:50]}...",
                            content=tweet['text'],
                            url=f"https://twitter.com/i/status/{tweet['id']}",
                            source='Twitter',
                            published_date=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            categories=['social', 'sentiment'],
                            metadata={
                                'retweet_count': tweet['public_metrics']['retweet_count'],
                                'like_count': tweet['public_metrics']['like_count']
                            }
                        )
                        articles.append(article)
            except Exception as e:
                logger.error(f"Twitter fetch error: {e}")
        
        return articles
    
    async def fetch_reddit_crypto(self, subreddits: List[str] = None) -> List[FetchedArticle]:
        """Reddit'ten kripto postları çek"""
        
        import praw
        
        if not self.reddit_client_id:
            logger.warning("Reddit credentials not configured")
            return []
        
        articles = []
        
        if not subreddits:
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'defi']
        
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_secret,
                user_agent='KriptoHTSAI/1.0'
            )
            
            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=25):
                    article = FetchedArticle(
                        title=post.title,
                        content=post.selftext or post.title,
                        url=f"https://reddit.com{post.permalink}",
                        source=f'Reddit r/{sub_name}',
                        author=str(post.author),
                        published_date=datetime.fromtimestamp(post.created_utc),
                        categories=['social', 'sentiment'],
                        metadata={
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments
                        }
                    )
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
        
        return articles

class RegulatorFetcher:
    """Düzenleyici kurum haberlerini çeken sınıf"""
    
    def __init__(self):
        self.sources = {
            'sec': {
                'url': 'https://www.sec.gov/news/pressreleases',
                'rss': 'https://www.sec.gov/rss/news/press.xml',
                'keywords': ['crypto', 'bitcoin', 'digital asset', 'token']
            },
            'cftc': {
                'url': 'https://www.cftc.gov/PressRoom/PressReleases',
                'rss': 'https://www.cftc.gov/RSS/PressReleases',
                'keywords': ['crypto', 'bitcoin', 'digital', 'virtual currency']
            },
            'treasury': {
                'url': 'https://home.treasury.gov/news/press-releases',
                'keywords': ['crypto', 'digital asset', 'virtual currency']
            },
            'fed': {
                'url': 'https://www.federalreserve.gov/newsevents/pressreleases.htm',
                'rss': 'https://www.federalreserve.gov/feeds/press_all.xml',
                'keywords': ['crypto', 'digital currency', 'cbdc']
            },
            'ecb': {
                'url': 'https://www.ecb.europa.eu/press/pr/html/index.en.html',
                'rss': 'https://www.ecb.europa.eu/rss/press.html',
                'keywords': ['crypto', 'digital euro', 'virtual asset']
            }
        }
    
    async def fetch_regulatory_news(self) -> List[FetchedArticle]:
        """Tüm düzenleyici kurumlardan haber çek"""
        
        articles = []
        
        for regulator, config in self.sources.items():
            # Try RSS first
            if 'rss' in config:
                rss_articles = await self._fetch_regulator_rss(regulator, config['rss'])
                articles.extend(rss_articles)
            
            # Scrape if needed
            if config.get('scrape', False):
                scraped = await self._scrape_regulator(regulator, config['url'])
                articles.extend(scraped)
        
        # Filter for crypto-related content
        crypto_articles = self._filter_crypto_news(articles)
        
        return crypto_articles
    
    async def _fetch_regulator_rss(self, regulator: str, rss_url: str) -> List[FetchedArticle]:
        """Düzenleyici kurum RSS feed'i"""
        
        articles = []
        
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:20]:
                article = FetchedArticle(
                    title=entry.title,
                    content=entry.get('summary', entry.get('description', '')),
                    url=entry.link,
                    source=f'{regulator.upper()} Press Release',
                    published_date=datetime.fromtimestamp(time.mktime(entry.published_parsed)),
                    categories=['regulation', 'official'],
                    metadata={'regulator': regulator}
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Regulator RSS error ({regulator}): {e}")
        
        return articles
    
    def _filter_crypto_news(self, articles: List[FetchedArticle]) -> List[FetchedArticle]:
        """Kripto ilgili haberleri filtrele"""
        
        filtered = []
        
        crypto_keywords = [
            'crypto', 'bitcoin', 'ethereum', 'digital asset',
            'virtual currency', 'blockchain', 'defi', 'stablecoin',
            'cbdc', 'token', 'nft'
        ]
        
        for article in articles:
            content_lower = (article.title + ' ' + article.content).lower()
            
            if any(keyword in content_lower for keyword in crypto_keywords):
                filtered.append(article)
        
        return filtered

# ============================================
# DATA ENRICHMENT
# ============================================

class DataEnricher:
    """Çekilen verileri zenginleştirme"""
    
    def __init__(self):
        self.translator = None  # For translation
        self.summarizer = None  # For summarization
    
    async def enrich_articles(self, articles: List[FetchedArticle]) -> List[FetchedArticle]:
        """Makaleleri zenginleştir"""
        
        enriched = []
        
        for article in articles:
            # Add summary if missing
            if not article.summary and article.content:
                article.summary = self._generate_summary(article.content)
            
            # Translate if needed
            if article.language != 'en':
                translated = await self._translate_article(article)
                if translated:
                    article = translated
            
            # Extract entities
            article.metadata['entities'] = self._extract_entities(article.content)
            
            # Calculate readability
            article.metadata['readability'] = self._calculate_readability(article.content)
            
            enriched.append(article)
        
        return enriched
    
    def _generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """Otomatik özet oluştur"""
        
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            
            parser = PlaintextParser.from_string(content, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, max_sentences)
            
            return ' '.join([str(sentence) for sentence in summary])
        except:
            # Simple fallback
            sentences = content.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    async def _translate_article(self, article: FetchedArticle) -> Optional[FetchedArticle]:
        """Makaleyi İngilizce'ye çevir"""
        
        try:
            from googletrans import Translator
            
            if not self.translator:
                self.translator = Translator()
            
            # Translate title and content
            title_trans = self.translator.translate(article.title, dest='en')
            content_trans = self.translator.translate(article.content[:5000], dest='en')
            
            article.title = title_trans.text
            article.content = content_trans.text
            article.metadata['original_language'] = article.language
            article.language = 'en'
            
            return article
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None
    
    def _extract_entities(self, content: str) -> List[str]:
        """Named entities çıkar"""
        
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(content[:10000])  # Limit for performance
            
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY']:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_
                    })
            
            return entities
            
        except:
            return []
    
    def _calculate_readability(self, content: str) -> float:
        """Okunabilirlik skoru hesapla"""
        
        try:
            from textstat import flesch_reading_ease
            return flesch_reading_ease(content)
        except:
            return 0

# ============================================
# SCHEDULER VE MANAGER
# ============================================

class DataFetcherManager:
    """Data fetcher yönetimi ve zamanlama"""
    
    def __init__(self):
        self.fetcher = EnhancedDataFetcher()
        self.social_fetcher = SocialMediaFetcher()
        self.regulator_fetcher = RegulatorFetcher()
        self.enricher = DataEnricher()
        self.storage = DataStorage()
        
    async def run_scheduled_fetch(self):
        """Zamanlanmış veri çekme"""
        
        while True:
            try:
                # High priority sources - every 5 minutes
                high_priority = await self.fetcher.fetch_all(priority_min=4)
                
                # Social media - every 10 minutes
                social = await self.social_fetcher.fetch_twitter_crypto()
                social.extend(await self.social_fetcher.fetch_reddit_crypto())
                
                # Regulatory - every hour
                regulatory = await self.regulator_fetcher.fetch_regulatory_news()
                
                # Combine all
                all_articles = high_priority + social + regulatory
                
                # Enrich
                enriched = await self.enricher.enrich_articles(all_articles)
                
                # Store
                await self.storage.store_articles(enriched)
                
                logger.info(f"Fetched and stored {len(enriched)} articles")
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Scheduled fetch error: {e}")
                await asyncio.sleep(60)

class DataStorage:
    """Veri depolama"""
    
    def __init__(self, db_path: str = 'news_data.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Database başlat"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                hash_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                url TEXT,
                source TEXT,
                author TEXT,
                published_date TIMESTAMP,
                categories TEXT,
                tags TEXT,
                sentiment REAL,
                language TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_articles(self, articles: List[FetchedArticle]):
        """Makaleleri veritabanına kaydet"""
        
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO articles
                    (hash_id, title, content, url, source, author, 
                     published_date, categories, tags, sentiment, 
                     language, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.hash_id,
                    article.title,
                    article.content,
                    article.url,
                    article.source,
                    article.author,
                    article.published_date,
                    json.dumps(article.categories),
                    json.dumps(article.tags),
                    article.sentiment,
                    article.language,
                    json.dumps(article.metadata)
                ))
            except Exception as e:
                logger.error(f"Storage error: {e}")
        
        conn.commit()
        conn.close()
        """Generic API response parsing"""