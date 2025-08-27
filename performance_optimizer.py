import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cachetools
from functools import lru_cache, wraps
import time
import redis
import pickle
from typing import Any, Dict, List, Optional
import aioredis

class PerformanceOptimizer:
    """Sistem performansını optimize eden sınıf"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.cache = cachetools.TTLCache(maxsize=1000, ttl=300)  # 5 dakika TTL
        self.redis_client = redis.from_url(redis_url)
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
    def cache_result(self, ttl: int = 300):
        """Sonuç önbellekleme dekoratörü"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Redis'te kontrol et
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
                
                # Fonksiyonu çalıştır
                result = await func(*args, **kwargs)
                
                # Redis'e kaydet
                self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    pickle.dumps(result)
                )
                
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                if cache_key in self.cache:
                    return self.cache[cache_key]
                
                result = func(*args, **kwargs)
                self.cache[cache_key] = result
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    async def batch_process(self, items: List, process_func, batch_size: int = 10):
        """Toplu işleme için optimize edilmiş fonksiyon"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Paralel işleme
            tasks = [process_func(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Rate limiting için küçük bekleme
            await asyncio.sleep(0.1)
        
        return results

# ==============================================
# database_models.py
# SQLAlchemy ORM modelleri
# ==============================================

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

class NewsArticle(Base):
    """Haber makaleleri tablosu"""
    __tablename__ = 'news_articles'
    
    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text)
    url = Column(String)
    published_at = Column(DateTime, nullable=False)
    coins_mentioned = Column(JSON)
    sentiment_score = Column(Float)
    sentiment_type = Column(String)
    impact_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class MarketMetric(Base):
    """Piyasa metrikleri tablosu"""
    __tablename__ = 'market_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coin = Column(String, nullable=False)
    price = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    price_change_24h = Column(Float)
    volatility = Column(Float)
    rsi = Column(Float)
    macd_data = Column(JSON)
    bollinger_data = Column(JSON)
    support_levels = Column(JSON)
    resistance_levels = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
class TradingSignal(Base):
    """İşlem sinyalleri tablosu"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coin = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)
    confidence = Column(Float)
    risk_level = Column(String)
    entry_price = Column(Float)
    target_prices = Column(JSON)
    stop_loss = Column(Float)
    risk_reward_ratio = Column(Float)
    strategy_type = Column(String)
    leverage = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Performans takibi
    actual_outcome = Column(JSON, nullable=True)
    performance_score = Column(Float, nullable=True)
    evaluated_at = Column(DateTime, nullable=True)

class ModelPerformance(Base):
    """Model performans metrikleri"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coin = Column(String, nullable=False)
    model_version = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    total_predictions = Column(Integer)
    successful_predictions = Column(Integer)
    training_date = Column(DateTime)
    evaluation_date = Column(DateTime, default=datetime.utcnow)

# ==============================================
# advanced_strategies.py
# Gelişmiş ticaret stratejileri
# ==============================================

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class StrategyType(Enum):
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"
    DCA = "dca"  # Dollar Cost Averaging

class AdvancedTradingStrategies:
    """Gelişmiş ticaret stratejileri sınıfı"""
    
    def __init__(self):
        self.strategies = {
            StrategyType.SCALPING: self._scalping_strategy,
            StrategyType.SWING_TRADING: self._swing_trading_strategy,
            StrategyType.POSITION_TRADING: self._position_trading_strategy,
            StrategyType.GRID_TRADING: self._grid_trading_strategy,
            StrategyType.DCA: self._dca_strategy
        }
    
    def generate_strategy(self, 
                         strategy_type: StrategyType,
                         market_data: Dict,
                         risk_profile: str = "moderate") -> Dict:
        """Strateji üret"""
        
        if strategy_type not in self.strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_func = self.strategies[strategy_type]
        return strategy_func(market_data, risk_profile)
    
    def _scalping_strategy(self, market_data: Dict, risk_profile: str) -> Dict:
        """Scalping stratejisi - Küçük fiyat hareketlerinden kar"""
        
        price = market_data['price']
        volatility = market_data['volatility']
        volume = market_data['volume_24h']
        
        # Yüksek hacim ve düşük volatilite ideal
        if volume < 1000000 or volatility > 0.05:
            return {'action': 'wait', 'reason': 'Unsuitable market conditions'}
        
        # Küçük hedefler, sıkı stoplar
        entry_price = price * 1.001
        take_profit = price * 1.003  # %0.3 kar
        stop_loss = price * 0.998    # %0.2 zarar
        
        return {
            'strategy': 'scalping',
            'action': 'buy',
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'position_size': self._calculate_position_size(risk_profile, 'scalping'),
            'timeframe': '1m-5m',
            'indicators': ['order_book_imbalance', 'bid_ask_spread', 'volume_profile']
        }
    
    def _swing_trading_strategy(self, market_data: Dict, risk_profile: str) -> Dict:
        """Swing trading - Orta vadeli trend takibi"""
        
        price = market_data['price']
        rsi = market_data['rsi']
        macd = market_data['macd']
        
        # Oversold veya overbought durumları ara
        if rsi < 30 and macd['histogram'] > 0:
            action = 'buy'
            target_multiplier = 1.05  # %5 hedef
        elif rsi > 70 and macd['histogram'] < 0:
            action = 'sell'
            target_multiplier = 0.95
        else:
            return {'action': 'wait', 'reason': 'No clear swing setup'}
        
        return {
            'strategy': 'swing_trading',
            'action': action,
            'entry_price': price,
            'target_price': price * target_multiplier,
            'stop_loss': price * (0.97 if action == 'buy' else 1.03),
            'holding_period': '2-7 days',
            'position_size': self._calculate_position_size(risk_profile, 'swing'),
            'indicators': ['rsi', 'macd', 'support_resistance', 'volume']
        }
    
    def _position_trading_strategy(self, market_data: Dict, risk_profile: str) -> Dict:
        """Pozisyon ticareti - Uzun vadeli trend takibi"""
        
        # Haftalık ve aylık trendleri analiz et
        price = market_data['price']
        ma_50 = market_data.get('ma_50', price)
        ma_200 = market_data.get('ma_200', price)
        
        # Golden cross / Death cross
        if ma_50 > ma_200 * 1.01:  # Golden cross
            signal = 'strong_buy'
            target = price * 1.20  # %20 uzun vadeli hedef
        elif ma_50 < ma_200 * 0.99:  # Death cross
            signal = 'strong_sell'
            target = price * 0.80
        else:
            signal = 'hold'
            target = price
        
        return {
            'strategy': 'position_trading',
            'signal': signal,
            'entry_zones': self._calculate_entry_zones(price, signal),
            'target_price': target,
            'stop_loss': price * (0.85 if 'buy' in signal else 1.15),
            'holding_period': 'weeks to months',
            'position_size': self._calculate_position_size(risk_profile, 'position'),
            'rebalance_frequency': 'weekly'
        }
    
    def _grid_trading_strategy(self, market_data: Dict, risk_profile: str) -> Dict:
        """Grid trading - Yatay piyasalar için"""
        
        price = market_data['price']
        volatility = market_data['volatility']
        support = market_data['support_levels']
        resistance = market_data['resistance_levels']
        
        if not support or not resistance:
            return {'action': 'wait', 'reason': 'Insufficient price levels'}
        
        # Grid seviyelerini hesapla
        grid_levels = self._calculate_grid_levels(
            lower_bound=min(support),
            upper_bound=max(resistance),
            num_grids=10
        )
        
        return {
            'strategy': 'grid_trading',
            'grid_levels': grid_levels,
            'order_size_per_grid': 100,  # USD
            'total_investment': 1000,
            'expected_profit_per_grid': volatility * 100,
            'risk_level': risk_profile,
            'auto_compound': True
        }
    
    def _dca_strategy(self, market_data: Dict, risk_profile: str) -> Dict:
        """Dollar Cost Averaging stratejisi"""
        
        price = market_data['price']
        trend = market_data.get('trend', 'neutral')
        
        # DCA parametreleri
        if trend == 'bearish':
            frequency = 'daily'
            amount_multiplier = 1.2  # Düşüşte daha fazla al
        elif trend == 'bullish':
            frequency = 'weekly'
            amount_multiplier = 0.8  # Yükselişte daha az al
        else:
            frequency = 'every_3_days'
            amount_multiplier = 1.0
        
        return {
            'strategy': 'dca',
            'base_amount': 100,  # USD
            'frequency': frequency,
            'amount_multiplier': amount_multiplier,
            'buy_conditions': {
                'max_price': price * 1.10,
                'min_volume': 500000,
                'max_rsi': 70
            },
            'exit_strategy': {
                'target_profit': 0.30,  # %30
                'max_holding_period': '6_months',
                'stop_loss': -0.20  # %20 zarar
            }
        }
    
    def _calculate_position_size(self, risk_profile: str, strategy: str) -> float:
        """Risk profiline göre pozisyon büyüklüğü hesapla"""
        
        base_sizes = {
            'conservative': {'scalping': 100, 'swing': 200, 'position': 500},
            'moderate': {'scalping': 200, 'swing': 500, 'position': 1000},
            'aggressive': {'scalping': 500, 'swing': 1000, 'position': 2000}
        }
        
        return base_sizes.get(risk_profile, base_sizes['moderate']).get(strategy, 100)
    
    def _calculate_entry_zones(self, price: float, signal: str) -> List[float]:
        """Kademeli giriş bölgelerini hesapla"""
        
        if 'buy' in signal:
            zones = [
                price * 0.98,  # İlk alım
                price * 0.96,  # İkinci alım
                price * 0.94   # Üçüncü alım
            ]
        else:
            zones = [
                price * 1.02,
                price * 1.04,
                price * 1.06
            ]
        
        return zones
    
    def _calculate_grid_levels(self, lower_bound: float, 
                              upper_bound: float, 
                              num_grids: int) -> List[Dict]:
        """Grid seviyelerini hesapla"""
        
        price_range = upper_bound - lower_bound
        grid_size = price_range / num_grids
        
        levels = []
        for i in range(num_grids + 1):
            level_price = lower_bound + (grid_size * i)
            levels.append({
                'price': level_price,
                'action': 'buy' if i < num_grids / 2 else 'sell',
                'grid_number': i,
                'profit_target': grid_size * 0.5
            })
        
        return levels

# ==============================================
# backtest_engine.py
# Geriye dönük test motoru
# ==============================================

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class BacktestEngine:
    """Strateji geriye dönük test motoru"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades_history = []
        self.performance_metrics = {}
        
    def run_backtest(self, 
                    strategy_func, 
                    historical_data: pd.DataFrame,
                    start_date: datetime,
                    end_date: datetime) -> Dict:
        """Geriye dönük testi çalıştır"""
        
        # Veriyi filtrele
        test_data = historical_data[
            (historical_data.index >= start_date) & 
            (historical_data.index <= end_date)
        ]
        
        # Her zaman dilimi için stratejiyi uygula
        for index, row in test_data.iterrows():
            market_data = row.to_dict()
            signal = strategy_func(market_data)
            
            if signal['action'] == 'buy':
                self._open_position(signal, index)
            elif signal['action'] == 'sell':
                self._close_position(signal, index)
            
            # Pozisyonları güncelle
            self._update_positions(row['close'])
        
        # Performans metriklerini hesapla
        self._calculate_performance()
        
        return self.performance_metrics
    
    def _open_position(self, signal: Dict, timestamp: datetime):
        """Pozisyon aç"""
        
        position_size = min(signal.get('position_size', 100), self.capital)
        
        if position_size > 0:
            position = {
                'timestamp': timestamp,
                'type': 'buy',
                'price': signal['entry_price'],
                'size': position_size,
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'strategy': signal.get('strategy', 'unknown')
            }
            
            self.positions.append(position)
            self.capital -= position_size
    
    def _close_position(self, signal: Dict, timestamp: datetime):
        """Pozisyon kapat"""
        
        if not self.positions:
            return
        
        position = self.positions.pop(0)  # FIFO
        
        exit_price = signal.get('exit_price', signal.get('entry_price'))
        pnl = (exit_price - position['price']) * (position['size'] / position['price'])
        
        trade = {
            'open_time': position['timestamp'],
            'close_time': timestamp,
            'entry_price': position['price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'return': pnl / position['size'],
            'strategy': position['strategy']
        }
        
        self.trades_history.append(trade)
        self.capital += position['size'] + pnl
    
    def _update_positions(self, current_price: float):
        """Açık pozisyonları güncelle"""
        
        for position in self.positions[:]:
            # Stop loss kontrolü
            if position['stop_loss'] and current_price <= position['stop_loss']:
                self._force_close(position, current_price, 'stop_loss')
            
            # Take profit kontrolü
            elif position['take_profit'] and current_price >= position['take_profit']:
                self._force_close(position, current_price, 'take_profit')
    
    def _force_close(self, position: Dict, price: float, reason: str):
        """Pozisyonu zorla kapat"""
        
        self.positions.remove(position)
        pnl = (price - position['price']) * (position['size'] / position['price'])
        
        trade = {
            'open_time': position['timestamp'],
            'close_time': datetime.now(),
            'entry_price': position['price'],
            'exit_price': price,
            'size': position['size'],
            'pnl': pnl,
            'return': pnl / position['size'],
            'strategy': position['strategy'],
            'close_reason': reason
        }
        
        self.trades_history.append(trade)
        self.capital += position['size'] + pnl
    
    def _calculate_performance(self):
        """Performans metriklerini hesapla"""
        
        if not self.trades_history:
            return
        
        trades_df = pd.DataFrame(self.trades_history)
        
        # Temel metrikler
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Kar/Zarar metrikleri
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrikleri
        returns = trades_df['return']
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(trades_df)
        
        self.performance_metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'avg_trade_duration': self._calculate_avg_duration(trades_df)
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Sharpe oranını hesapla"""
        
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Günlük risk-free rate
        
        if excess_returns.std() == 0:
            return 0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Maksimum drawdown hesapla"""
        
        cumulative_returns = (1 + trades_df['return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return drawdowns.min()
    
    def _calculate_avg_duration(self, trades_df: pd.DataFrame) -> float:
        """Ortalama işlem süresini hesapla"""
        
        durations = []
        for _, trade in trades_df.iterrows():
            duration = (trade['close_time'] - trade['open_time']).total_seconds() / 3600  # Saat
            durations.append(duration)
        
        return np.mean(durations) if durations else 0