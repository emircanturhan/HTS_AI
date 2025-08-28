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

# ===== ESKİ KÜTÜPHANE TAMAMEN KALDIRILDI =====
# ===== YENİ, MODERN KÜTÜPHANE KULLANILIYOR =====
import pandas_ta as ta

# Veritabanı
import sqlite3
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Bildirim sistemleri
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters

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
            # Diğer kaynaklar için metodlar eklenecek
        except Exception as e:
            logger.error(f"Haber çekme hatası ({source}): {e}")
            return []
    
    async def _fetch_coindesk_news(self) -> List[NewsItem]:
        # ... (Bu fonksiyonun içeriği aynı kalabilir) ...
        pass
    
    async def _fetch_twitter_posts(self) -> List[NewsItem]:
        # ... (Bu fonksiyonun içeriği aynı kalabilir) ...
        pass
    
    async def fetch_market_data(self, symbol: str) -> Optional[CoinMetrics]:
        """Belirtilen coin için piyasa verilerini çek"""
        try:
            ticker_url = f"{self.sources['binance']}/ticker/24hr?symbol={symbol}USDT"
            klines_url = f"{self.sources['binance']}/klines?symbol={symbol}USDT&interval=1h&limit=100"
            
            async with self.session.get(ticker_url) as response:
                ticker_data = await response.json()
            
            async with self.session.get(klines_url) as response:
                klines_data = await response.json()

            df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])

            # Teknik göstergeleri PANDAS-TA ile hesapla
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(append=True)

            support_resistance = self._calculate_support_resistance(df)
            
            if df.empty or 'RSI_14' not in df.columns:
                logger.warning(f"Teknik analiz verileri {symbol} için üretilemedi.")
                return None

            return CoinMetrics(
                symbol=symbol,
                price=float(ticker_data['lastPrice']),
                volume_24h=float(ticker_data['volume']),
                market_cap=0,
                price_change_24h=float(ticker_data['priceChangePercent']),
                volatility=df['close'].pct_change().rolling(24).std().iloc[-1],
                rsi=df['RSI_14'].iloc[-1],
                macd={'macd': df['MACD_12_26_9'].iloc[-1], 'signal': df['MACDs_12_26_9'].iloc[-1], 'histogram': df['MACDh_12_26_9'].iloc[-1]},
                bollinger_bands={'upper': df['BBU_20_2.0'].iloc[-1], 'middle': df['BBM_20_2.0'].iloc[-1], 'lower': df['BBL_20_2.0'].iloc[-1]},
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance']
            )
        except Exception as e:
            logger.error(f"Piyasa verisi çekme hatası ({symbol}): {e}")
            return None
    
    def _extract_coin_mentions(self, text: str) -> List[str]:
        # ... (Bu fonksiyonun içeriği aynı kalabilir) ...
        pass
    
    def _calculate_support_resistance(self, df: pd.DataFrame):
        # ... (Bu fonksiyonun içeriği aynı kalabilir) ...
        pass

# ============================================
# DUYGU ANALİZİ MODÜLÜ
# ============================================

class SentimentAnalyzer:
    # ... (Bu sınıfın içeriği aynı kalabilir) ...
    pass
    
# ============================================
# TAHMİN VE SİNYAL MODÜLÜ
# ============================================

class PredictionEngine:
    # ... (Bu sınıfın içeriği aynı kalabilir) ...
    pass
    
# ============================================
# BİLDİRİM SİSTEMİ
# ============================================

class NotificationManager:
    # ... (Bu sınıfın içeriğini yeni telegram-bot kütüphanesine göre güncelleyeceğiz) ...
    def __init__(self, telegram_token: str, chat_id: str):
        self.application = Application.builder().token(telegram_token).build()
        self.chat_id = chat_id
        # ... (geri kalan kod aynı) ...
    async def send_message(self, message: str):
        await self.application.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')

    # send_signal ve send_news_alert fonksiyonları yukarıdaki send_message'ı kullanacak şekilde güncellenebilir.
    # Örnek:
    async def send_signal(self, signal: MarketSignal):
        emoji = self._get_signal_emoji(signal.signal_type)
        message = f"..." # Mesaj içeriği önceki gibi
        await self.send_message(message)

# ============================================
# ANA UYGULAMA
# ============================================

class KriptoHTSAI:
    # ... (Bu sınıfın içeriği çoğunlukla aynı, sadece telegram kısmı değişebilir) ...
    pass
    
# ============================================
# GERİ BESLEME VE ÖĞRENME
# ============================================

class FeedbackLoop:
    # ... (Bu sınıfın içeriği aynı kalabilir) ...
    pass

# ============================================
# YAPAY ZEKA YORUMLAYICI
# ============================================

class AIInterpreter:
    # ... (Bu sınıfın içeriği aynı kalabilir) ...
    pass

# ============================================
# ANA ÇALIŞTIRMA
# ============================================

async def main():
    # ... (Bu fonksiyonun içeriği aynı kalabilir) ...
    pass

if __name__ == "__main__":
    # NLTK verilerini indir
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    
    # Ana döngüyü başlat
    asyncio.run(main())