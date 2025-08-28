import ccxt
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import json
from enum import Enum
import time

logger = logging.getLogger(__name__)

# ============================================
# TRADING ENUMS VE DATA STRUCTURES
# ============================================

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class TradingOrder:
    """Trading order veri yapısı"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    amount: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled: float = 0
    remaining: float = 0
    timestamp: datetime = None
    exchange: str = "binance"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.remaining == 0:
            self.remaining = self.amount

@dataclass
class Position:
    """Açık pozisyon veri yapısı"""
    symbol: str
    side: PositionSide
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    opened_at: datetime = None
    
    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()

# ============================================
# AUTOMATED TRADING EXECUTOR
# ============================================

class AutomatedTradingExecutor:
    """Otomatik trading execution sistemi"""
    
    def __init__(self, exchange_configs: Dict):
        self.exchanges = {}
        self.positions = {}
        self.orders = {}
        self.balance = {}
        
        # Risk parametreleri
        self.risk_params = {
            'max_position_size': 1000,  # USD
            'max_open_positions': 5,
            'max_daily_trades': 20,
            'max_slippage': 0.01,  # %1
            'min_volume_threshold': 100000,  # $100k daily volume
            'emergency_stop': False
        }
        
        # Initialize exchanges
        self._initialize_exchanges(exchange_configs)
        
        # Trading statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'daily_trades': 0,
            'last_reset': datetime.now()
        }
    
    def _initialize_exchanges(self, configs: Dict):
        """Exchange bağlantılarını başlat"""
        for exchange_name, config in configs.items():
            try:
                if exchange_name == 'binance':
                    self.exchanges[exchange_name] = ccxt.binance({
                        'apiKey': config['api_key'],
                        'secret': config['secret'],
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future' if config.get('futures', False) else 'spot'
                        }
                    })
                elif exchange_name == 'bybit':
                    self.exchanges[exchange_name] = ccxt.bybit({
                        'apiKey': config['api_key'],
                        'secret': config['secret'],
                        'enableRateLimit': True
                    })
                # Add more exchanges as needed
                
                logger.info(f"Exchange {exchange_name} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")
    
    async def execute_signal(self, signal: Dict) -> Dict:
        """Trading sinyalini execute et"""
        try:
            # Pre-execution checks
            if not await self._pre_execution_checks(signal):
                return {'status': 'rejected', 'reason': 'Pre-execution checks failed'}
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            
            # Smart order routing
            exchange = await self._select_best_exchange(signal['symbol'])
            
            # Create and execute order
            order = await self._create_order(
                exchange=exchange,
                symbol=signal['symbol'],
                side=signal['side'],
                amount=position_size,
                order_type=signal.get('order_type', OrderType.MARKET),
                price=signal.get('entry_price'),
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit')
            )
            
            # Monitor order execution
            await self._monitor_order_execution(order)
            
            # Update statistics
            self._update_statistics(order)
            
            return {
                'status': 'success',
                'order': order,
                'exchange': exchange
            }
            
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _pre_execution_checks(self, signal: Dict) -> bool:
        """Execution öncesi kontroller"""
        
        # Emergency stop check
        if self.risk_params['emergency_stop']:
            logger.warning("Emergency stop is active")
            return False
        
        # Daily trade limit check
        if self.stats['daily_trades'] >= self.risk_params['max_daily_trades']:
            logger.warning("Daily trade limit reached")
            return False
        
        # Open positions limit check
        if len(self.positions) >= self.risk_params['max_open_positions']:
            logger.warning("Maximum open positions reached")
            return False
        
        # Volume check
        volume = await self._get_24h_volume(signal['symbol'])
        if volume < self.risk_params['min_volume_threshold']:
            logger.warning(f"Insufficient volume for {signal['symbol']}")
            return False
        
        # Duplicate position check
        if signal['symbol'] in self.positions:
            logger.warning(f"Position already exists for {signal['symbol']}")
            return False
        
        return True
    
    async def _calculate_position_size(self, signal: Dict) -> float:
        """Position size hesaplama"""
        
        # Get account balance
        balance = await self._get_account_balance()
        
        # Risk-based sizing
        risk_amount = balance * 0.02  # %2 risk per trade
        
        # Signal confidence adjustment
        confidence = signal.get('confidence', 0.5)
        adjusted_size = risk_amount * confidence
        
        # Apply limits
        position_size = min(
            adjusted_size,
            self.risk_params['max_position_size'],
            balance * 0.1  # Max %10 of balance
        )
        
        # Round to exchange requirements
        return self._round_to_precision(position_size, signal['symbol'])
    
    async def _select_best_exchange(self, symbol: str) -> str:
        """En iyi exchange'i seç"""
        
        best_exchange = None
        best_price = float('inf')
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = exchange.fetch_ticker(symbol)
                
                # Check liquidity and spread
                spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                
                if spread < self.risk_params['max_slippage']:
                    if ticker['ask'] < best_price:
                        best_price = ticker['ask']
                        best_exchange = exchange_name
                        
            except:
                continue
        
        return best_exchange or 'binance'  # Default to binance
    
    async def _create_order(self, **kwargs) -> TradingOrder:
        """Order oluştur ve gönder"""
        
        exchange = self.exchanges[kwargs['exchange']]
        
        # Create main order
        if kwargs['order_type'] == OrderType.MARKET:
            order_result = exchange.create_market_order(
                kwargs['symbol'],
                kwargs['side'],
                kwargs['amount']
            )
        elif kwargs['order_type'] == OrderType.LIMIT:
            order_result = exchange.create_limit_order(
                kwargs['symbol'],
                kwargs['side'],
                kwargs['amount'],
                kwargs['price']
            )
        
        # Create stop-loss if specified
        if kwargs.get('stop_loss'):
            stop_order = exchange.create_order(
                kwargs['symbol'],
                'stop_loss',
                'sell' if kwargs['side'] == 'buy' else 'buy',
                kwargs['amount'],
                None,
                {'stopPrice': kwargs['stop_loss']}
            )
        
        # Create take-profit if specified
        if kwargs.get('take_profit'):
            tp_order = exchange.create_limit_order(
                kwargs['symbol'],
                'sell' if kwargs['side'] == 'buy' else 'buy',
                kwargs['amount'],
                kwargs['take_profit']
            )
        
        # Create TradingOrder object
        order = TradingOrder(
            order_id=order_result['id'],
            symbol=kwargs['symbol'],
            side=kwargs['side'],
            order_type=kwargs['order_type'],
            amount=kwargs['amount'],
            price=kwargs.get('price'),
            stop_price=kwargs.get('stop_loss'),
            status=OrderStatus.OPEN,
            exchange=kwargs['exchange']
        )
        
        # Store order
        self.orders[order.order_id] = order
        
        return order
    
    async def _monitor_order_execution(self, order: TradingOrder):
        """Order execution'ı monitor et"""
        
        exchange = self.exchanges[order.exchange]
        max_wait_time = 60  # 60 seconds
        start_time = time.time()
        
        while order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            try:
                # Check order status
                order_info = exchange.fetch_order(order.order_id, order.symbol)
                
                # Update order status
                order.status = self._map_order_status(order_info['status'])
                order.filled = order_info['filled']
                order.remaining = order_info['remaining']
                
                # Check for partial fills
                if order.filled > 0 and order.status != OrderStatus.FILLED:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                # Timeout check
                if time.time() - start_time > max_wait_time:
                    if order.status == OrderStatus.OPEN:
                        # Cancel the order
                        await self._cancel_order(order)
                    break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring order {order.order_id}: {e}")
                break
        
        # Create position if filled
        if order.status == OrderStatus.FILLED:
            await self._create_position(order)
    
    def _map_order_status(self, exchange_status: str) -> OrderStatus:
        """Exchange status'ü OrderStatus'a map et"""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED
        }
        return status_map.get(exchange_status, OrderStatus.PENDING)
    
    async def _create_position(self, order: TradingOrder):
        """Filled order'dan pozisyon oluştur"""
        
        position = Position(
            symbol=order.symbol,
            side=PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT,
            amount=order.filled,
            entry_price=order.price or await self._get_current_price(order.symbol),
            current_price=await self._get_current_price(order.symbol),
            unrealized_pnl=0,
            stop_loss=order.stop_price
        )
        
        self.positions[order.symbol] = position
        logger.info(f"Position created: {position}")
    
    async def _cancel_order(self, order: TradingOrder):
        """Order'ı iptal et"""
        try:
            exchange = self.exchanges[order.exchange]
            exchange.cancel_order(order.order_id, order.symbol)
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order {order.order_id} cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
    
    async def manage_positions(self):
        """Açık pozisyonları yönet"""
        
        while True:
            try:
                for symbol, position in list(self.positions.items()):
                    # Update current price
                    position.current_price = await self._get_current_price(symbol)
                    
                    # Calculate P&L
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.amount
                    
                    # Check stop-loss
                    if position.stop_loss:
                        if (position.side == PositionSide.LONG and position.current_price <= position.stop_loss) or \
                           (position.side == PositionSide.SHORT and position.current_price >= position.stop_loss):
                            await self._close_position(symbol, "Stop-loss triggered")
                    
                    # Check take-profit
                    if position.take_profit:
                        if (position.side == PositionSide.LONG and position.current_price >= position.take_profit) or \
                           (position.side == PositionSide.SHORT and position.current_price <= position.take_profit):
                            await self._close_position(symbol, "Take-profit triggered")
                    
                    # Trailing stop-loss
                    if position.trailing_stop_distance:
                        await self._update_trailing_stop(position)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Position management error: {e}")
                await asyncio.sleep(10)
    
    async def _update_trailing_stop(self, position: Position):
        """Trailing stop-loss güncelle"""
        
        if position.side == PositionSide.LONG:
            new_stop = position.current_price - position.trailing_stop_distance
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.info(f"Trailing stop updated for {position.symbol}: {new_stop}")
        else:
            new_stop = position.current_price + position.trailing_stop_distance
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.info(f"Trailing stop updated for {position.symbol}: {new_stop}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Pozisyonu kapat"""
        
        position = self.positions.get(symbol)
        if not position:
            return
        
        try:
            # Create market order to close
            exchange = self.exchanges['binance']  # Use primary exchange
            
            close_side = 'sell' if position.side == PositionSide.LONG else 'buy'
            
            order_result = exchange.create_market_order(
                symbol,
                close_side,
                position.amount
            )
            
            # Update position
            position.realized_pnl = position.unrealized_pnl
            
            # Update statistics
            if position.realized_pnl > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            
            self.stats['total_pnl'] += position.realized_pnl
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Position closed for {symbol}: {reason}, P&L: {position.realized_pnl}")
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Güncel fiyatı al"""
        try:
            exchange = self.exchanges['binance']
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except:
            return 0
    
    async def _get_24h_volume(self, symbol: str) -> float:
        """24 saatlik hacmi al"""
        try:
            exchange = self.exchanges['binance']
            ticker = exchange.fetch_ticker(symbol)
            return ticker['quoteVolume']
        except:
            return 0
    
    async def _get_account_balance(self) -> float:
        """Hesap bakiyesini al"""
        try:
            exchange = self.exchanges['binance']
            balance = exchange.fetch_balance()
            return balance['USDT']['free']
        except:
            return 0
    
    def _round_to_precision(self, amount: float, symbol: str) -> float:
        """Exchange precision'a göre yuvarla"""
        # Simplified - in production would get from exchange.markets
        precision_map = {
            'BTC/USDT': 0.001,
            'ETH/USDT': 0.01,
            'BNB/USDT': 0.1
        }
        precision = precision_map.get(symbol, 0.01)
        return round(amount / precision) * precision
    
    def _update_statistics(self, order: TradingOrder):
        """İstatistikleri güncelle"""
        self.stats['total_trades'] += 1
        self.stats['daily_trades'] += 1
        
        # Reset daily counter
        if datetime.now().date() > self.stats['last_reset'].date():
            self.stats['daily_trades'] = 0
            self.stats['last_reset'] = datetime.now()
    
    async def emergency_stop_all(self):
        """Acil durum - tüm pozisyonları kapat"""
        logger.warning("EMERGENCY STOP TRIGGERED - Closing all positions")
        self.risk_params['emergency_stop'] = True
        
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol, "Emergency stop")
        
        # Cancel all open orders
        for order_id, order in list(self.orders.items()):
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                await self._cancel_order(order)
    
    def get_performance_stats(self) -> Dict:
        """Performans istatistiklerini al"""
        
        total_trades = self.stats['total_trades']
        if total_trades == 0:
            return {}
        
        win_rate = self.stats['winning_trades'] / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': win_rate,
            'total_pnl': self.stats['total_pnl'],
            'avg_pnl_per_trade': self.stats['total_pnl'] / total_trades,
            'daily_trades': self.stats['daily_trades'],
            'open_positions': len(self.positions),
            'total_exposure': sum(p.amount * p.current_price for p in self.positions.values())
        }

# ============================================
# SMART ORDER ROUTER
# ============================================

class SmartOrderRouter:
    """Akıllı order routing - best execution"""
    
    def __init__(self, exchanges: Dict):
        self.exchanges = exchanges
        self.routing_history = []
        
    async def route_order(self, symbol: str, side: str, 
                         amount: float, order_type: OrderType) -> Dict:
        """Order'ı en iyi şekilde route et"""
        
        # Get order books from all exchanges
        order_books = await self._get_order_books(symbol)
        
        # Find best execution venue
        if order_type == OrderType.MARKET:
            best_venue = await self._find_best_market_execution(
                order_books, side, amount
            )
        else:
            best_venue = await self._find_best_limit_execution(
                order_books, side, amount
            )
        
        # Check for order splitting opportunity
        if amount > best_venue['max_executable']:
            # Split order across multiple venues
            split_orders = await self._split_order(
                symbol, side, amount, order_books
            )
            return {'type': 'split', 'orders': split_orders}
        
        return {'type': 'single', 'venue': best_venue}
    
    async def _get_order_books(self, symbol: str) -> Dict:
        """Tüm exchange'lerden order book al"""
        order_books = {}
        
        for name, exchange in self.exchanges.items():
            try:
                book = exchange.fetch_order_book(symbol)
                order_books[name] = book
            except:
                continue
        
        return order_books
    
    async def _find_best_market_execution(self, order_books: Dict, 
                                         side: str, amount: float) -> Dict:
        """Market order için en iyi venue bul"""
        
        best_price = float('inf') if side == 'buy' else 0
        best_venue = None
        
        for exchange_name, book in order_books.items():
            # Calculate average execution price
            if side == 'buy':
                avg_price = self._calculate_avg_price(book['asks'], amount)
                if avg_price < best_price:
                    best_price = avg_price
                    best_venue = exchange_name
            else:
                avg_price = self._calculate_avg_price(book['bids'], amount)
                if avg_price > best_price:
                    best_price = avg_price
                    best_venue = exchange_name
        
        return {
            'exchange': best_venue,
            'expected_price': best_price,
            'max_executable': self._get_max_executable(order_books[best_venue], side)
        }
    
    def _calculate_avg_price(self, orders: List, amount: float) -> float:
        """Ortalama execution price hesapla"""
        
        total_cost = 0
        total_amount = 0
        
        for price, size in orders:
            if total_amount >= amount:
                break
            
            executable = min(size, amount - total_amount)
            total_cost += price * executable
            total_amount += executable
        
        return total_cost / total_amount if total_amount > 0 else 0
    
    def _get_max_executable(self, order_book: Dict, side: str) -> float:
        """Maximum executable amount hesapla"""
        
        orders = order_book['asks'] if side == 'buy' else order_book['bids']
        return sum(size for price, size in orders[:10])  # Top 10 levels
    
    async def _split_order(self, symbol: str, side: str, 
                          amount: float, order_books: Dict) -> List[Dict]:
        """Order'ı multiple venue'ye böl"""
        
        split_orders = []
        remaining = amount
        
        # Sort venues by best price
        venues = []
        for exchange_name, book in order_books.items():
            if side == 'buy':
                best_price = book['asks'][0][0] if book['asks'] else float('inf')
            else:
                best_price = book['bids'][0][0] if book['bids'] else 0
            
            venues.append((exchange_name, best_price))
        
        venues.sort(key=lambda x: x[1], reverse=(side == 'sell'))
        
        # Allocate to venues
        for exchange_name, price in venues:
            if remaining <= 0:
                break
            
            max_exec = self._get_max_executable(order_books[exchange_name], side)
            allocation = min(remaining, max_exec)
            
            if allocation > 0:
                split_orders.append({
                    'exchange': exchange_name,
                    'amount': allocation,
                    'expected_price': price
                })
                remaining -= allocation
        
        return split_orders

# ============================================
# EXECUTION ALGORITHMS
# ============================================

class ExecutionAlgorithms:
    """Gelişmiş execution algoritmaları"""
    
    def __init__(self, executor: AutomatedTradingExecutor):
        self.executor = executor
        
    async def twap_execution(self, symbol: str, side: str, 
                            total_amount: float, duration_minutes: int):
        """Time-Weighted Average Price (TWAP) execution"""
        
        slices = duration_minutes  # One order per minute
        slice_amount = total_amount / slices
        
        results = []
        
        for i in range(slices):
            # Execute slice
            order = await self.executor.execute_signal({
                'symbol': symbol,
                'side': side,
                'amount': slice_amount,
                'order_type': OrderType.MARKET
            })
            
            results.append(order)
            
            # Wait for next slice
            await asyncio.sleep(60)
        
        return results
    
    async def vwap_execution(self, symbol: str, side: str, 
                           total_amount: float, duration_minutes: int):
        """Volume-Weighted Average Price (VWAP) execution"""
        
        # Get historical volume pattern
        volume_profile = await self._get_volume_profile(symbol)
        
        results = []
        
        for minute in range(duration_minutes):
            # Calculate slice based on volume profile
            volume_weight = volume_profile.get(minute % 60, 1.0)
            slice_amount = total_amount * volume_weight / sum(volume_profile.values())
            
            if slice_amount > 0:
                order = await self.executor.execute_signal({
                    'symbol': symbol,
                    'side': side,
                    'amount': slice_amount,
                    'order_type': OrderType.MARKET
                })
                
                results.append(order)
            
            await asyncio.sleep(60)
        
        return results
    
    async def iceberg_execution(self, symbol: str, side: str,
                              total_amount: float, visible_amount: float):
        """Iceberg order - sadece bir kısmı görünür"""
        
        results = []
        remaining = total_amount
        
        while remaining > 0:
            # Show only visible amount
            current_slice = min(visible_amount, remaining)
            
            order = await self.executor.execute_signal({
                'symbol': symbol,
                'side': side,
                'amount': current_slice,
                'order_type': OrderType.LIMIT,
                'price': await self._get_best_limit_price(symbol, side)
            })
            
            results.append(order)
            remaining -= current_slice
            
            # Random delay to avoid detection
            await asyncio.sleep(np.random.uniform(5, 15))
        
        return results
    
    async def _get_volume_profile(self, symbol: str) -> Dict:
        """Volume profile al"""
        # Simplified - in production would use historical data
        return {i: np.random.uniform(0.5, 1.5) for i in range(60)}
    
    async def _get_best_limit_price(self, symbol: str, side: str) -> float:
        """En iyi limit price hesapla"""
        price = await self.executor._get_current_price(symbol)
        
        # Place slightly better than market
        if side == 'buy':
            return price * 0.999
        else:
            return price * 1.001