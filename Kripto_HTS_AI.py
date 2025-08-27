import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Temel kütüphaneler
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Web scraping ve API
import requests
from bs4 import BeautifulSoup
import aiohttp

# NLP ve Sentiment Analysis
from textblob import TextBlob
import nltk
from transformers import pipeline

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Teknik Analiz
import talib

# Veritabanı
import sqlite3
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Bildirim sistemleri
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Konfigürasyon yükle
load_dotenv()

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kripto_hts_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# VERİ YAPILARI VE ENUM'LAR
# ============================================

class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    STRONG_POSITIVE = "strong_positive"
    STRONG_NEGATIVE = "strong_negative"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class NewsItem:
    """Haber öğesi veri yapısı"""
    id: str
    source: str
    title: str
    content: str
    url: str
    published_at: datetime
    coins_mentioned: List[str]
    sentiment_score: float = 0.0
    sentiment_type: SentimentType = SentimentType.NEUTRAL
    impact_score: float = 0.0
    
@dataclass
class MarketSignal:
    """Piyasa sinyali veri yapısı"""
    coin: str
    signal_type: SignalType
    confidence: float
    risk_level: RiskLevel
    entry_price: float
    target_prices: List[float]
    stop_loss: float
    risk_reward_ratio: float
    strategy_type: str  # "spot" veya "margin"
    leverage: Optional[int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class CoinMetrics:
    """Coin metrikleri veri yapısı"""
    symbol: str
    price: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    volatility: float
    rsi: float
    macd: Dict
    bollinger_bands: Dict
    support_levels: List[float]
    resistance_levels: List[float]

# ============================================
# VERİ TOPLAMA MODÜLÜ
# ============================================

class DataCollector:
    """Çoklu kaynaklardan veri toplama sınıfı"""
    
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.sources = {
            'coindesk': 'https://www.coindesk.com/api/v1/latest',
            'cointelegraph': 'https://cointelegraph.com/api/news',
            'cryptonews': 'https://cryptonews.com/api/news',
            'binance': 'https://api.binance.com/api/v3',
            'coingecko': 'https://api.coingecko.com/api/v3',
            'twitter': 'https://api.twitter.com/2',
            'sec': 'https://www.sec.gov/news/pressreleases',
            'fed': 'https://www.federalreserve.gov/feeds/press_all.xml'
        }
        self.api_keys = {
            'binance': os.getenv('BINANCE_API_KEY'),
            'twitter': os.getenv('TWITTER_BEARER_TOKEN'),
            'coingecko': os.getenv('COINGECKO_API_KEY')
        }
        
    async def fetch_news(self, source: str) -> List[NewsItem]:
        """Belirtilen kaynaktan haberleri çek"""
        try:
            if source == 'coindesk':
                return await self._fetch_coindesk_news()
            elif source == 'twitter':
                return await self._fetch_twitter_posts()
            elif source == 'sec':
                return await self._fetch_sec_announcements()
            # Diğer kaynaklar için metodlar eklenecek
        except Exception as e:
            logger.error(f"Haber çekme hatası ({source}): {e}")
            return []
    
    async def _fetch_coindesk_news(self) -> List[NewsItem]:
        """CoinDesk haberlerini çek"""
        news_items = []
        try:
            async with self.session.get(self.sources['coindesk']) as response:
                data = await response.json()
                for article in data.get('articles', []):
                    news_item = NewsItem(
                        id=article['id'],
                        source='coindesk',
                        title=article['title'],
                        content=article['content'],
                        url=article['url'],
                        published_at=datetime.fromisoformat(article['published_at']),
                        coins_mentioned=self._extract_coin_mentions(article['content'])
                    )
                    news_items.append(news_item)
        except Exception as e:
            logger.error(f"CoinDesk API hatası: {e}")
        return news_items
    
    async def _fetch_twitter_posts(self) -> List[NewsItem]:
        """Önemli kripto Twitter hesaplarından gönderileri çek"""
        important_accounts = [
            'VitalikButerin', 'elonmusk', 'saylor', 'APompliano',
            'CZ_Binance', 'brian_armstrong', 'SBF_FTX'
        ]
        news_items = []
        headers = {'Authorization': f"Bearer {self.api_keys['twitter']}"}
        
        for account in important_accounts:
            try:
                url = f"{self.sources['twitter']}/users/by/username/{account}/tweets"
                async with self.session.get(url, headers=headers) as response:
                    data = await response.json()
                    # Tweet'leri işle
                    pass
            except Exception as e:
                logger.error(f"Twitter API hatası ({account}): {e}")
        return news_items
    
    async def fetch_market_data(self, symbol: str) -> CoinMetrics:
        """Belirtilen coin için piyasa verilerini çek"""
        try:
            # Binance'den fiyat ve hacim verileri
            ticker_url = f"{self.sources['binance']}/ticker/24hr?symbol={symbol}USDT"
            klines_url = f"{self.sources['binance']}/klines?symbol={symbol}USDT&interval=1h&limit=100"
            
            async with self.session.get(ticker_url) as response:
                ticker_data = await response.json()
            
            async with self.session.get(klines_url) as response:
                klines_data = await response.json()
            
            # Teknik göstergeleri hesapla
            closes = np.array([float(k[4]) for k in klines_data])
            highs = np.array([float(k[2]) for k in klines_data])
            lows = np.array([float(k[3]) for k in klines_data])
            volumes = np.array([float(k[5]) for k in klines_data])
            
            # RSI hesaplama
            rsi = talib.RSI(closes, timeperiod=14)[-1]
            
            # MACD hesaplama
            macd, signal, hist = talib.MACD(closes)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(closes)
            
            # Destek ve direnç seviyeleri
            support_resistance = self._calculate_support_resistance(highs, lows, closes)
            
            return CoinMetrics(
                symbol=symbol,
                price=float(ticker_data['lastPrice']),
                volume_24h=float(ticker_data['volume']),
                market_cap=0,  # CoinGecko'dan alınacak
                price_change_24h=float(ticker_data['priceChangePercent']),
                volatility=np.std(closes[-24:]),
                rsi=rsi,
                macd={'macd': macd[-1], 'signal': signal[-1], 'histogram': hist[-1]},
                bollinger_bands={'upper': upper[-1], 'middle': middle[-1], 'lower': lower[-1]},
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance']
            )
        except Exception as e:
            logger.error(f"Piyasa verisi çekme hatası ({symbol}): {e}")
            return None
    
    def _extract_coin_mentions(self, text: str) -> List[str]:
        """Metinden bahsedilen coinleri çıkar"""
        coin_patterns = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC']
        mentioned = []
        text_upper = text.upper()
        for coin in coin_patterns:
            if coin in text_upper:
                mentioned.append(coin)
        return mentioned
    
    def _calculate_support_resistance(self, highs, lows, closes):
        """Destek ve direnç seviyelerini hesapla"""
        # Basit pivot noktası hesaplama
        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
        r1 = 2 * pivot - lows[-1]
        r2 = pivot + (highs[-1] - lows[-1])
        s1 = 2 * pivot - highs[-1]
        s2 = pivot - (highs[-1] - lows[-1])
        
        return {
            'support': [s2, s1],
            'resistance': [r1, r2]
        }

# ============================================
# DUYGU ANALİZİ MODÜLÜ
# ============================================

class SentimentAnalyzer:
    """Gelişmiş duygu analizi sınıfı"""
    
    def __init__(self):
        # Transformer tabanlı model yükle
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"  # Finansal metinler için özel model
        )
        
        # Özel anahtar kelimeler ve ağırlıkları
        self.keywords = {
            'positive': {
                'bullish': 2.0, 'moon': 1.5, 'pump': 1.5, 'breakout': 1.8,
                'adoption': 1.6, 'partnership': 1.7, 'upgrade': 1.5,
                'accumulate': 1.4, 'support': 1.2, 'recovery': 1.3
            },
            'negative': {
                'bearish': -2.0, 'crash': -2.5, 'dump': -2.0, 'scam': -3.0,
                'hack': -2.8, 'regulation': -1.5, 'ban': -2.5, 'sell': -1.5,
                'resistance': -1.2, 'correction': -1.3
            }
        }
        
    def analyze(self, news_item: NewsItem) -> Tuple[float, SentimentType]:
        """Haber öğesinin duygu analizini yap"""
        try:
            # Başlık ve içerik analizini birleştir
            title_sentiment = self._analyze_text(news_item.title, weight=1.5)
            content_sentiment = self._analyze_text(news_item.content, weight=1.0)
            
            # Anahtar kelime analizi
            keyword_score = self._keyword_analysis(news_item.content)
            
            # Toplam skor hesaplama
            total_score = (title_sentiment * 1.5 + content_sentiment + keyword_score) / 3.5
            
            # Sentiment tipi belirleme
            if total_score >= 0.6:
                sentiment_type = SentimentType.STRONG_POSITIVE
            elif total_score >= 0.2:
                sentiment_type = SentimentType.POSITIVE
            elif total_score <= -0.6:
                sentiment_type = SentimentType.STRONG_NEGATIVE
            elif total_score <= -0.2:
                sentiment_type = SentimentType.NEGATIVE
            else:
                sentiment_type = SentimentType.NEUTRAL
                
            return total_score, sentiment_type
            
        except Exception as e:
            logger.error(f"Duygu analizi hatası: {e}")
            return 0.0, SentimentType.NEUTRAL
    
    def _analyze_text(self, text: str, weight: float = 1.0) -> float:
        """Metni analiz et ve skorla"""
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # BERT token limiti
            
            if result['label'] == 'POSITIVE':
                return result['score'] * weight
            elif result['label'] == 'NEGATIVE':
                return -result['score'] * weight
            else:
                return 0.0
        except:
            # Yedek olarak TextBlob kullan
            blob = TextBlob(text)
            return blob.sentiment.polarity * weight
    
    def _keyword_analysis(self, text: str) -> float:
        """Anahtar kelime bazlı analiz"""
        score = 0.0
        text_lower = text.lower()
        
        for word, weight in self.keywords['positive'].items():
            if word in text_lower:
                score += weight
                
        for word, weight in self.keywords['negative'].items():
            if word in text_lower:
                score += weight
                
        return np.tanh(score / 10)  # Skoru normalize et

# ============================================
# TAHMİN VE SİNYAL MODÜLÜ
# ============================================

class PredictionEngine:
    """Makine öğrenmesi tabanlı tahmin motoru"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'sentiment_score', 'volume_change', 'price_change',
            'rsi', 'macd_histogram', 'volatility', 'market_cap_rank',
            'twitter_mentions', 'news_frequency', 'whale_activity'
        ]
        
    def train_model(self, coin: str, training_data: pd.DataFrame):
        """Belirli bir coin için model eğit"""
        try:
            # Özellikleri hazırla
            X = training_data[self.feature_columns]
            y = training_data['price_direction']  # 1: yukarı, 0: aşağı
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Ölçeklendir
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Model oluştur ve eğit
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Modeli kaydet
            self.models[coin] = model
            
            # Performans metrikleri
            accuracy = model.score(X_test_scaled, y_test)
            logger.info(f"{coin} modeli eğitildi. Doğruluk: {accuracy:.2%}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Model eğitim hatası ({coin}): {e}")
            return 0.0
    
    def predict(self, coin: str, features: Dict) -> MarketSignal:
        """Tahmin yap ve sinyal üret"""
        try:
            if coin not in self.models:
                logger.warning(f"{coin} için model bulunamadı")
                return None
            
            # Özellikleri hazırla
            feature_vector = pd.DataFrame([features])[self.feature_columns]
            feature_scaled = self.scaler.transform(feature_vector)
            
            # Tahmin yap
            model = self.models[coin]
            prediction = model.predict(feature_scaled)[0]
            confidence = model.predict_proba(feature_scaled)[0].max()
            
            # Risk seviyesi belirle
            risk_level = self._calculate_risk_level(features)
            
            # Sinyal tipi belirle
            if prediction == 1 and confidence > 0.7:
                signal_type = SignalType.STRONG_BUY if confidence > 0.85 else SignalType.BUY
            elif prediction == 0 and confidence > 0.7:
                signal_type = SignalType.STRONG_SELL if confidence > 0.85 else SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Giriş, hedef ve stop seviyeleri hesapla
            current_price = features['current_price']
            entry_price, targets, stop_loss = self._calculate_levels(
                current_price, signal_type, features['volatility']
            )
            
            # Risk/Ödül oranı
            risk_reward = abs(targets[0] - entry_price) / abs(entry_price - stop_loss)
            
            return MarketSignal(
                coin=coin,
                signal_type=signal_type,
                confidence=confidence,
                risk_level=risk_level,
                entry_price=entry_price,
                target_prices=targets,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward,
                strategy_type="spot" if risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM] else "margin",
                leverage=2 if risk_level == RiskLevel.HIGH else None
            )
            
        except Exception as e:
            logger.error(f"Tahmin hatası ({coin}): {e}")
            return None
    
    def _calculate_risk_level(self, features: Dict) -> RiskLevel:
        """Risk seviyesini hesapla"""
        volatility = features.get('volatility', 0)
        volume_change = features.get('volume_change', 0)
        
        risk_score = (volatility * 0.5 + abs(volume_change) * 0.3 + 
                     (1 - features.get('rsi', 50) / 100) * 0.2)
        
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _calculate_levels(self, price: float, signal: SignalType, volatility: float):
        """Giriş, hedef ve stop seviyelerini hesapla"""
        vol_multiplier = max(volatility, 0.02)  # Min %2 volatilite
        
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            entry = price * 1.002  # %0.2 yukarıdan giriş
            targets = [
                price * (1 + vol_multiplier * 1.5),
                price * (1 + vol_multiplier * 2.5),
                price * (1 + vol_multiplier * 4)
            ]
            stop_loss = price * (1 - vol_multiplier * 0.8)
        else:
            entry = price * 0.998  # %0.2 aşağıdan giriş
            targets = [
                price * (1 - vol_multiplier * 1.5),
                price * (1 - vol_multiplier * 2.5),
                price * (1 - vol_multiplier * 4)
            ]
            stop_loss = price * (1 + vol_multiplier * 0.8)
            
        return entry, targets, stop_loss

# ============================================
# BİLDİRİM SİSTEMİ
# ============================================

class NotificationManager:
    """Telegram ve diğer bildirim sistemleri yöneticisi"""
    
    def __init__(self, telegram_token: str, chat_id: str):
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.alert_levels = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '📊',
            'low': 'ℹ️'
        }
        
    async def send_signal(self, signal: MarketSignal):
        """İşlem sinyali gönder"""
        emoji = self._get_signal_emoji(signal.signal_type)
        
        message = f"""
{emoji} **{signal.coin} İŞLEM SİNYALİ**

📈 Sinyal: {signal.signal_type.value.upper()}
💪 Güven: {signal.confidence:.1%}
⚡ Risk: {signal.risk_level.value.upper()}

💰 Giriş Fiyatı: ${signal.entry_price:.4f}
🎯 Hedefler:
  • T1: ${signal.target_prices[0]:.4f}
  • T2: ${signal.target_prices[1]:.4f}
  • T3: ${signal.target_prices[2]:.4f}
🛑 Stop Loss: ${signal.stop_loss:.4f}

📊 Risk/Ödül: {signal.risk_reward_ratio:.2f}
💼 Strateji: {signal.strategy_type.upper()}
{f"🔥 Kaldıraç: {signal.leverage}x" if signal.leverage else ""}

⏰ {signal.timestamp.strftime('%H:%M:%S')}
"""
        
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode='Markdown'
        )
    
    async def send_news_alert(self, news: NewsItem, impact: str = 'medium'):
        """Önemli haber bildirimi gönder"""
        emoji = self.alert_levels.get(impact, 'ℹ️')
        sentiment_emoji = self._get_sentiment_emoji(news.sentiment_type)
        
        message = f"""
{emoji} **HABER BİLDİRİMİ**

📰 {news.title}

{sentiment_emoji} Duygu: {news.sentiment_type.value.upper()} ({news.sentiment_score:.2f})
💥 Etki Skoru: {news.impact_score:.2f}
🪙 İlgili Coinler: {', '.join(news.coins_mentioned)}

🔗 [Detaylar]({news.url})

📅 {news.published_at.strftime('%H:%M - %d/%m/%Y')}
"""
        
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
    
    def _get_signal_emoji(self, signal_type: SignalType) -> str:
        """Sinyal tipine göre emoji seç"""
        emojis = {
            SignalType.STRONG_BUY: '🚀',
            SignalType.BUY: '📈',
            SignalType.HOLD: '⏸️',
            SignalType.SELL: '📉',
            SignalType.STRONG_SELL: '🔻'
        }
        return emojis.get(signal_type, '📊')
    
    def _get_sentiment_emoji(self, sentiment: SentimentType) -> str:
        """Duygu tipine göre emoji seç"""
        emojis = {
            SentimentType.STRONG_POSITIVE: '🟢',
            SentimentType.POSITIVE: '🟡',
            SentimentType.NEUTRAL: '⚪',
            SentimentType.NEGATIVE: '🟠',
            SentimentType.STRONG_NEGATIVE: '🔴'
        }
        return emojis.get(sentiment, '⚪')

# ============================================
# ANA UYGULAMA
# ============================================

class KriptoHTSAI:
    """Ana uygulama sınıfı"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.prediction_engine = PredictionEngine()
        self.notification_manager = NotificationManager(
            telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )
        
        # İzlenecek coinler (Binance Top 100)
        self.watchlist = self._load_watchlist()
        
        # Veritabanı bağlantısı
        self.db_engine = create_engine('sqlite:///kripto_hts_ai.db')
        self.Session = sessionmaker(bind=self.db_engine)
        
        # Çalışma durumu
        self.is_running = False
        
    def _load_watchlist(self) -> List[str]:
        """İzlenecek coin listesini yükle"""
        # Başlangıç için popüler coinler
        return ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 
                'MATIC', 'LINK', 'UNI', 'ATOM', 'FTM', 'NEAR', 'ALGO']
    
    async def start(self):
        """Ana döngüyü başlat"""
        self.is_running = True
        logger.info("Kripto_HTS_AI başlatılıyor...")
        
        # Başlangıç görevleri
        await self._initialize_models()
        
        # Ana döngü
        while self.is_running:
            try:
                # Paralel görevler
                tasks = [
                    self._news_monitoring_cycle(),
                    self._market_monitoring_cycle(),
                    self._signal_generation_cycle()
                ]
                
                await asyncio.gather(*tasks)
                
                # Döngü aralığı (5 dakika)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Ana döngü hatası: {e}")
                await asyncio.sleep(60)
    
    async def _initialize_models(self):
        """Modelleri başlat ve eğit"""
        logger.info("Modeller başlatılıyor...")
        
        for coin in self.watchlist:
            # Geçmiş verileri yükle (veritabanından veya API'den)
            # training_data = self._load_training_data(coin)
            # self.prediction_engine.train_model(coin, training_data)
            pass
    
    async def _news_monitoring_cycle(self):
        """Haber takip döngüsü"""
        sources = ['coindesk', 'twitter', 'sec']
        
        for source in sources:
            news_items = await self.data_collector.fetch_news(source)
            
            for news in news_items:
                # Duygu analizi
                score, sentiment = self.sentiment_analyzer.analyze(news)
                news.sentiment_score = score
                news.sentiment_type = sentiment
                
                # Etki analizi
                news.impact_score = self._calculate_impact(news)
                
                # Önemli haberleri bildir
                if abs(news.impact_score) > 0.7:
                    impact_level = 'high' if abs(news.impact_score) > 0.85 else 'medium'
                    await self.notification_manager.send_news_alert(news, impact_level)
                
                # Veritabanına kaydet
                self._save_news(news)
    
    async def _market_monitoring_cycle(self):
        """Piyasa takip döngüsü"""
        for coin in self.watchlist:
            metrics = await self.data_collector.fetch_market_data(coin)
            
            if metrics:
                # Anormal hareketleri tespit et
                if self._detect_anomaly(metrics):
                    logger.info(f"Anormal hareket tespit edildi: {coin}")
                    # Acil analiz tetikle
                    await self._emergency_analysis(coin, metrics)
                
                # Metrikleri kaydet
                self._save_metrics(metrics)
    
    async def _signal_generation_cycle(self):
        """Sinyal üretim döngüsü"""
        for coin in self.watchlist:
            try:
                # Son verileri topla
                features = await self._prepare_features(coin)
                
                if features:
                    # Tahmin yap
                    signal = self.prediction_engine.predict(coin, features)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        # Sinyali değerlendir ve bildir
                        if self._validate_signal(signal):
                            await self.notification_manager.send_signal(signal)
                            self._save_signal(signal)
                            
            except Exception as e:
                logger.error(f"Sinyal üretim hatası ({coin}): {e}")
    
    async def _prepare_features(self, coin: str) -> Dict:
        """ML modeli için özellikleri hazırla"""
        try:
            # Son 24 saatlik verileri al
            session = self.Session()
            
            # Piyasa metrikleri
            metrics = await self.data_collector.fetch_market_data(coin)
            
            # Son haberlerin ortalama duygu skoru
            recent_news_sentiment = self._get_recent_news_sentiment(coin, hours=24)
            
            # Sosyal medya metrikleri
            twitter_data = await self._get_social_metrics(coin)
            
            # Balina hareketleri (büyük işlemler)
            whale_activity = await self._detect_whale_movements(coin)
            
            features = {
                'current_price': metrics.price,
                'sentiment_score': recent_news_sentiment,
                'volume_change': metrics.volume_24h,
                'price_change': metrics.price_change_24h,
                'rsi': metrics.rsi,
                'macd_histogram': metrics.macd['histogram'],
                'volatility': metrics.volatility,
                'market_cap_rank': 0,  # TODO: CoinGecko'dan al
                'twitter_mentions': twitter_data.get('mentions', 0),
                'news_frequency': twitter_data.get('frequency', 0),
                'whale_activity': whale_activity
            }
            
            session.close()
            return features
            
        except Exception as e:
            logger.error(f"Özellik hazırlama hatası ({coin}): {e}")
            return None
    
    def _detect_anomaly(self, metrics: CoinMetrics) -> bool:
        """Anormal piyasa hareketlerini tespit et"""
        anomalies = []
        
        # Yüksek volatilite
        if metrics.volatility > 0.1:  # %10'dan fazla
            anomalies.append('high_volatility')
        
        # Aşırı RSI değerleri
        if metrics.rsi > 80 or metrics.rsi < 20:
            anomalies.append('extreme_rsi')
        
        # Bollinger Band kırılması
        if metrics.price > metrics.bollinger_bands['upper'] * 1.02:
            anomalies.append('bb_breakout_up')
        elif metrics.price < metrics.bollinger_bands['lower'] * 0.98:
            anomalies.append('bb_breakout_down')
        
        # Yüksek hacim değişimi
        # TODO: Ortalama hacimle karşılaştır
        
        return len(anomalies) > 0
    
    async def _emergency_analysis(self, coin: str, metrics: CoinMetrics):
        """Acil durum analizi ve bildirim"""
        logger.warning(f"Acil analiz başlatılıyor: {coin}")
        
        # Hızlı tahmin
        features = await self._prepare_features(coin)
        if features:
            signal = self.prediction_engine.predict(coin, features)
            
            if signal and signal.confidence > 0.8:
                # Acil bildirim gönder
                signal.signal_type = SignalType.STRONG_BUY if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else SignalType.STRONG_SELL
                await self.notification_manager.send_signal(signal)
    
    def _calculate_impact(self, news: NewsItem) -> float:
        """Haberin piyasa etkisini hesapla"""
        impact = 0.0
        
        # Kaynak güvenilirliği
        source_weights = {
            'sec': 2.0,
            'coindesk': 1.5,
            'cointelegraph': 1.3,
            'twitter': 1.0
        }
        source_weight = source_weights.get(news.source, 1.0)
        
        # Bahsedilen coin sayısı
        coin_factor = min(len(news.coins_mentioned) * 0.2, 1.0)
        
        # Sentiment gücü
        sentiment_factor = abs(news.sentiment_score)
        
        # Toplam etki
        impact = (source_weight * sentiment_factor * (1 + coin_factor)) / 3
        
        return np.tanh(impact)  # -1 ile 1 arasında normalize et
    
    def _get_recent_news_sentiment(self, coin: str, hours: int = 24) -> float:
        """Son X saatteki haberlerin ortalama duygu skorunu al"""
        session = self.Session()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # TODO: Veritabanından ilgili haberleri çek ve ortalama al
        # query = session.query(NewsTable).filter(...)
        
        session.close()
        return 0.0  # Şimdilik placeholder
    
    async def _get_social_metrics(self, coin: str) -> Dict:
        """Sosyal medya metriklerini al"""
        # TODO: Twitter API entegrasyonu
        return {
            'mentions': 0,
            'frequency': 0,
            'sentiment': 0.0
        }
    
    async def _detect_whale_movements(self, coin: str) -> float:
        """Balina hareketlerini tespit et"""
        # TODO: Blockchain explorer API entegrasyonu
        # Büyük transferleri tespit et
        return 0.0
    
    def _validate_signal(self, signal: MarketSignal) -> bool:
        """Sinyali doğrula"""
        # Risk/Ödül kontrolü
        if signal.risk_reward_ratio < 1.5:
            logger.info(f"Sinyal reddedildi: Düşük risk/ödül oranı ({signal.coin})")
            return False
        
        # Güven kontrolü
        if signal.confidence < 0.65:
            logger.info(f"Sinyal reddedildi: Düşük güven ({signal.coin})")
            return False
        
        # Aşırı risk kontrolü
        if signal.risk_level == RiskLevel.EXTREME and signal.confidence < 0.9:
            logger.info(f"Sinyal reddedildi: Aşırı risk ({signal.coin})")
            return False
        
        return True
    
    def _save_news(self, news: NewsItem):
        """Haberi veritabanına kaydet"""
        session = self.Session()
        try:
            # TODO: NewsTable ORM modeli oluştur ve kaydet
            pass
        finally:
            session.close()
    
    def _save_metrics(self, metrics: CoinMetrics):
        """Metrikleri veritabanına kaydet"""
        session = self.Session()
        try:
            # TODO: MetricsTable ORM modeli oluştur ve kaydet
            pass
        finally:
            session.close()
    
    def _save_signal(self, signal: MarketSignal):
        """Sinyali veritabanına kaydet"""
        session = self.Session()
        try:
            # TODO: SignalTable ORM modeli oluştur ve kaydet
            pass
        finally:
            session.close()
    
    async def stop(self):
        """Uygulamayı durdur"""
        logger.info("Kripto_HTS_AI durduruluyor...")
        self.is_running = False
        await self.data_collector.session.close()

# ============================================
# GERİ BESLEME VE ÖĞRENME
# ============================================

class FeedbackLoop:
    """Otomatik öğrenme ve adaptasyon sistemi"""
    
    def __init__(self, prediction_engine: PredictionEngine):
        self.prediction_engine = prediction_engine
        self.performance_history = []
        
    def evaluate_signal(self, signal: MarketSignal, actual_outcome: Dict):
        """Sinyalin performansını değerlendir"""
        # Tahmin edilen yön ile gerçek yönü karşılaştır
        predicted_direction = 1 if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else -1
        actual_direction = 1 if actual_outcome['price_change'] > 0 else -1
        
        # Hedeflere ulaşma durumu
        targets_hit = 0
        for target in signal.target_prices:
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if actual_outcome['high_price'] >= target:
                    targets_hit += 1
            else:
                if actual_outcome['low_price'] <= target:
                    targets_hit += 1
        
        # Stop loss tetiklenme durumu
        stop_hit = False
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            if actual_outcome['low_price'] <= signal.stop_loss:
                stop_hit = True
        else:
            if actual_outcome['high_price'] >= signal.stop_loss:
                stop_hit = True
        
        # Performans skoru hesapla
        performance_score = self._calculate_performance_score(
            predicted_direction == actual_direction,
            targets_hit,
            stop_hit,
            signal.confidence
        )
        
        # Kaydet
        self.performance_history.append({
            'signal': signal,
            'outcome': actual_outcome,
            'score': performance_score,
            'timestamp': datetime.now()
        })
        
        # Model güncelleme gereksinimi kontrolü
        if len(self.performance_history) >= 100:
            self._trigger_model_update(signal.coin)
    
    def _calculate_performance_score(self, direction_correct: bool, 
                                    targets_hit: int, stop_hit: bool, 
                                    confidence: float) -> float:
        """Performans skoru hesapla"""
        score = 0.0
        
        # Yön tahmini doğruluğu
        if direction_correct:
            score += 40
        
        # Hedeflere ulaşma
        score += targets_hit * 20
        
        # Stop loss tetiklenmemesi
        if not stop_hit:
            score += 20
        
        # Güven faktörü
        if direction_correct:
            score *= confidence
        else:
            score *= (1 - confidence)
        
        return score / 100  # 0-1 arasında normalize et
    
    def _trigger_model_update(self, coin: str):
        """Model güncelleme tetikle"""
        logger.info(f"{coin} modeli için güncelleme başlatılıyor...")
        
        # Son performansları analiz et
        recent_performance = [p for p in self.performance_history[-100:] 
                             if p['signal'].coin == coin]
        
        avg_score = np.mean([p['score'] for p in recent_performance])
        
        if avg_score < 0.6:  # Performans düşükse
            logger.warning(f"{coin} modeli düşük performans gösteriyor: {avg_score:.2%}")
            # Yeni verilerle yeniden eğit
            # TODO: Implement retraining logic

# ============================================
# YAPAY ZEKA YORUMLAYICI
# ============================================

class AIInterpreter:
    """Karmaşık olayları yorumlayan AI modülü"""
    
    def __init__(self):
        self.context_window = []  # Son olayları sakla
        self.market_regime = 'neutral'  # bull, bear, neutral
        
    def interpret_macro_event(self, event: Dict) -> Dict:
        """Makro ekonomik olayları yorumla"""
        interpretations = {
            'fed_rate_hike': {
                'impact': 'negative',
                'magnitude': 'high',
                'affected_coins': ['BTC', 'ETH'],
                'reasoning': 'Faiz artışı risk varlıklarından çıkışa neden olur'
            },
            'btc_etf_approval': {
                'impact': 'positive',
                'magnitude': 'very_high',
                'affected_coins': ['BTC', 'ETH', 'ALL'],
                'reasoning': 'Kurumsal para girişini kolaylaştırır'
            },
            'stable_coin_depeg': {
                'impact': 'negative',
                'magnitude': 'extreme',
                'affected_coins': ['ALL'],
                'reasoning': 'Sistematik risk ve güven kaybı'
            }
        }
        
        event_type = event.get('type')
        if event_type in interpretations:
            return interpretations[event_type]
        
        # Bilinmeyen olay için AI analizi
        return self._analyze_unknown_event(event)
    
    def _analyze_unknown_event(self, event: Dict) -> Dict:
        """Bilinmeyen olayları analiz et"""
        # TODO: GPT veya benzeri model entegrasyonu
        return {
            'impact': 'unknown',
            'magnitude': 'medium',
            'affected_coins': [],
            'reasoning': 'Yeterli veri yok'
        }
    
    def detect_market_regime(self, metrics: List[CoinMetrics]) -> str:
        """Piyasa rejimini tespit et"""
        if not metrics:
            return 'neutral'
        
        # Ortalama performans hesapla
        avg_change = np.mean([m.price_change_24h for m in metrics])
        avg_volume = np.mean([m.volume_24h for m in metrics])
        
        # Rejim tespiti
        if avg_change > 5 and avg_volume > 1.2:  # %5 artış, %20 hacim artışı
            self.market_regime = 'bull'
        elif avg_change < -5 and avg_volume > 1.2:
            self.market_regime = 'bear'
        else:
            self.market_regime = 'neutral'
        
        return self.market_regime

# ============================================
# ANA ÇALIŞTIRMA
# ============================================

async def main():
    """Ana çalıştırma fonksiyonu"""
    # Ortam değişkenlerini kontrol et
    required_env = [
        'BINANCE_API_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing = [var for var in required_env if not os.getenv(var)]
    if missing:
        logger.error(f"Eksik ortam değişkenleri: {missing}")
        logger.info("Lütfen .env dosyanızı oluşturun ve gerekli API anahtarlarını ekleyin")
        return
    
    # Uygulamayı başlat
    app = KriptoHTSAI()
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu")
        await app.stop()
    except Exception as e:
        logger.error(f"Kritik hata: {e}")
        await app.stop()

if __name__ == "__main__":
    # NLTK verilerini indir
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    
    # Ana döngüyü başlat
    asyncio.run(main())