import praw
import discord
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from collections import defaultdict
import logging
from dataclasses import dataclass
from web3 import Web3
import requests

logger = logging.getLogger(__name__)

# ============================================
# SOCIAL SENTIMENT - REDDIT
# ============================================

class RedditSentimentAnalyzer:
    """Reddit kripto topluluklarından sentiment analizi"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # İzlenecek subreddit'ler
        self.crypto_subreddits = [
            'cryptocurrency',
            'bitcoin',
            'ethereum', 
            'defi',
            'cryptomarkets',
            'altcoin',
            'binance',
            'coinbase',
            'cryptomoonshots',
            'satoshistreetbets'
        ]
        
        # Coin pattern'leri
        self.coin_patterns = {
            'BTC': r'\b(bitcoin|btc|₿)\b',
            'ETH': r'\b(ethereum|eth|ether)\b',
            'BNB': r'\b(binance|bnb)\b',
            'ADA': r'\b(cardano|ada)\b',
            'SOL': r'\b(solana|sol)\b',
            'DOT': r'\b(polkadot|dot)\b',
            'AVAX': r'\b(avalanche|avax)\b',
            'MATIC': r'\b(polygon|matic)\b',
            'LINK': r'\b(chainlink|link)\b',
            'UNI': r'\b(uniswap|uni)\b'
        }
        
        # Sentiment weights for specific terms
        self.sentiment_weights = {
            'moon': 2.0,
            'lambo': 1.5,
            'bullish': 1.8,
            'pump': 1.5,
            'rocket': 1.5,
            'hodl': 1.2,
            'buy': 1.0,
            'long': 1.0,
            'bearish': -1.8,
            'dump': -2.0,
            'crash': -2.5,
            'scam': -3.0,
            'rug': -3.0,
            'sell': -1.0,
            'short': -1.5,
            'fud': -1.5
        }
    
    async def analyze_subreddit_sentiment(self, subreddit_name: str, 
                                         time_filter: str = 'hour',
                                         limit: int = 100) -> Dict:
        """Subreddit sentiment analizi"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            sentiment_data = defaultdict(lambda: {
                'mentions': 0,
                'sentiment_scores': [],
                'upvotes': 0,
                'comments': 0,
                'posts': []
            })
            
            # Hot posts
            for submission in subreddit.hot(limit=limit):
                # Skip stickied posts
                if submission.stickied:
                    continue
                
                # Analyze title and selftext
                full_text = f"{submission.title} {submission.selftext}".lower()
                
                # Find coin mentions
                mentioned_coins = self._extract_coin_mentions(full_text)
                
                # Calculate sentiment
                sentiment = self._calculate_weighted_sentiment(full_text)
                
                for coin in mentioned_coins:
                    sentiment_data[coin]['mentions'] += 1
                    sentiment_data[coin]['sentiment_scores'].append(sentiment)
                    sentiment_data[coin]['upvotes'] += submission.score
                    sentiment_data[coin]['comments'] += submission.num_comments
                    sentiment_data[coin]['posts'].append({
                        'title': submission.title,
                        'url': submission.url,
                        'score': submission.score,
                        'created': datetime.fromtimestamp(submission.created_utc)
                    })
                
                # Analyze top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]:  # Top 20 comments
                    comment_text = comment.body.lower()
                    comment_coins = self._extract_coin_mentions(comment_text)
                    comment_sentiment = self._calculate_weighted_sentiment(comment_text)
                    
                    for coin in comment_coins:
                        sentiment_data[coin]['sentiment_scores'].append(comment_sentiment)
                        sentiment_data[coin]['upvotes'] += comment.score
            
            # Calculate aggregate scores
            results = {}
            for coin, data in sentiment_data.items():
                if data['sentiment_scores']:
                    avg_sentiment = np.mean(data['sentiment_scores'])
                    weighted_sentiment = self._calculate_weighted_score(data)
                    
                    results[coin] = {
                        'subreddit': subreddit_name,
                        'mentions': data['mentions'],
                        'average_sentiment': avg_sentiment,
                        'weighted_sentiment': weighted_sentiment,
                        'total_upvotes': data['upvotes'],
                        'total_comments': data['comments'],
                        'trend': self._determine_trend(data['posts']),
                        'top_posts': data['posts'][:5],
                        'timestamp': datetime.now()
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Reddit analysis error: {e}")
            return {}
    
    def _extract_coin_mentions(self, text: str) -> List[str]:
        """Metinden coin isimlerini çıkar"""
        mentioned = []
        
        for coin, pattern in self.coin_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                mentioned.append(coin)
        
        return mentioned
    
    def _calculate_weighted_sentiment(self, text: str) -> float:
        """Ağırlıklı sentiment skoru hesapla"""
        # Base sentiment with TextBlob
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity
        
        # Apply keyword weights
        weighted_score = base_sentiment
        
        for keyword, weight in self.sentiment_weights.items():
            if keyword in text:
                weighted_score += weight * 0.1
        
        # Normalize to [-1, 1]
        return max(-1, min(1, weighted_score))
    
    def _calculate_weighted_score(self, data: Dict) -> float:
        """Upvote ve engagement bazlı ağırlıklı skor"""
        if not data['sentiment_scores']:
            return 0
        
        avg_sentiment = np.mean(data['sentiment_scores'])
        
        # Engagement factor
        engagement_factor = np.log1p(data['upvotes'] + data['comments'] * 2)
        
        # Weighted score
        weighted = avg_sentiment * (1 + engagement_factor / 10)
        
        return max(-1, min(1, weighted))
    
    def _determine_trend(self, posts: List[Dict]) -> str:
        """Mention trend'ini belirle"""
        if len(posts) < 2:
            return 'neutral'
        
        # Son 1 saat vs önceki 1 saat
        now = datetime.now()
        recent = [p for p in posts if (now - p['created']).total_seconds() < 3600]
        older = [p for p in posts if 3600 <= (now - p['created']).total_seconds() < 7200]
        
        if len(recent) > len(older) * 1.5:
            return 'increasing'
        elif len(recent) < len(older) * 0.7:
            return 'decreasing'
        else:
            return 'stable'
    
    async def monitor_realtime(self, callback):
        """Gerçek zamanlı Reddit monitoring"""
        while True:
            try:
                for subreddit_name in self.crypto_subreddits:
                    results = await self.analyze_subreddit_sentiment(
                        subreddit_name, 
                        time_filter='hour',
                        limit=50
                    )
                    
                    # Callback ile sonuçları işle
                    if results:
                        await callback(results)
                
                # 5 dakika bekle
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Realtime monitoring error: {e}")
                await asyncio.sleep(60)

# ============================================
# DISCORD SENTIMENT ANALYZER
# ============================================

class DiscordSentimentAnalyzer(discord.Client):
    """Discord kripto sunucularından sentiment analizi"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # İzlenecek Discord sunucuları (ID'ler)
        self.crypto_servers = {
            'bitcoin': 12345678,  # Örnek ID'ler
            'ethereum': 23456789,
            'defi': 34567890,
            'trading': 45678901
        }
        
        # Message buffer for analysis
        self.message_buffer = defaultdict(list)
        self.sentiment_cache = defaultdict(list)
        
    async def on_ready(self):
        """Bot hazır olduğunda"""
        logger.info(f'Discord bot logged in as {self.user}')
        
        # Start monitoring task
        self.loop.create_task(self.analyze_messages())
    
    async def on_message(self, message):
        """Yeni mesaj geldiğinde"""
        # Bot mesajlarını ignore et
        if message.author.bot:
            return
        
        # Kripto sunucularından gelen mesajları sakla
        if message.guild and message.guild.id in self.crypto_servers.values():
            self.message_buffer[message.guild.id].append({
                'content': message.content,
                'author': str(message.author),
                'channel': str(message.channel),
                'timestamp': message.created_at,
                'reactions': []
            })
    
    async def on_reaction_add(self, reaction, user):
        """Reaksiyon eklendiğinde"""
        # Sentiment indicator olarak kullan
        message_data = {
            'emoji': str(reaction.emoji),
            'count': reaction.count,
            'is_positive': self._is_positive_reaction(str(reaction.emoji))
        }
        
        # Buffer'daki mesajı güncelle
        # Implementation detail
    
    async def analyze_messages(self):
        """Periyodik mesaj analizi"""
        while not self.is_closed():
            try:
                await asyncio.sleep(300)  # 5 dakika
                
                for server_id, messages in self.message_buffer.items():
                    if messages:
                        sentiment = await self._analyze_server_sentiment(messages)
                        self.sentiment_cache[server_id].append(sentiment)
                        
                        # Buffer'ı temizle
                        self.message_buffer[server_id] = self.message_buffer[server_id][-100:]
                        
            except Exception as e:
                logger.error(f"Discord analysis error: {e}")
    
    async def _analyze_server_sentiment(self, messages: List[Dict]) -> Dict:
        """Server sentiment analizi"""
        coin_sentiments = defaultdict(list)
        
        for msg in messages:
            # Coin mentions
            mentioned_coins = self._extract_coins(msg['content'])
            
            # Sentiment
            sentiment = TextBlob(msg['content']).sentiment.polarity
            
            for coin in mentioned_coins:
                coin_sentiments[coin].append({
                    'sentiment': sentiment,
                    'timestamp': msg['timestamp'],
                    'channel': msg['channel']
                })
        
        # Aggregate results
        results = {}
        for coin, sentiments in coin_sentiments.items():
            if sentiments:
                results[coin] = {
                    'average_sentiment': np.mean([s['sentiment'] for s in sentiments]),
                    'message_count': len(sentiments),
                    'channels': list(set([s['channel'] for s in sentiments])),
                    'trend': self._calculate_trend(sentiments)
                }
        
        return results
    
    def _extract_coins(self, text: str) -> List[str]:
        """Extract coin symbols from text"""
        # Simple regex for common coins
        pattern = r'\b([A-Z]{3,5})\b'
        matches = re.findall(pattern, text.upper())
        
        # Filter to known coins
        known_coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC']
        return [coin for coin in matches if coin in known_coins]
    
    def _is_positive_reaction(self, emoji: str) -> bool:
        """Check if emoji is positive"""
        positive_emojis = ['👍', '🚀', '🔥', '💎', '🐂', '📈', '✅', '💚']
        return emoji in positive_emojis
    
    def _calculate_trend(self, sentiments: List[Dict]) -> str:
        """Calculate sentiment trend"""
        if len(sentiments) < 2:
            return 'neutral'
        
        # Recent vs older
        recent = sentiments[-len(sentiments)//2:]
        older = sentiments[:len(sentiments)//2]
        
        recent_avg = np.mean([s['sentiment'] for s in recent])
        older_avg = np.mean([s['sentiment'] for s in older])
        
        if recent_avg > older_avg * 1.1:
            return 'improving'
        elif recent_avg < older_avg * 0.9:
            return 'declining'
        else:
            return 'stable'

# ============================================
# WHALE TRACKING - ON-CHAIN ANALYSIS
# ============================================

@dataclass
class WhaleTransaction:
    """Whale transaction veri yapısı"""
    tx_hash: str
    from_address: str
    to_address: str
    token: str
    amount: float
    value_usd: float
    timestamp: datetime
    tx_type: str  # 'transfer', 'exchange_deposit', 'exchange_withdrawal'
    
class WhaleTracker:
    """Blockchain'de büyük transferleri takip et"""
    
    def __init__(self, eth_rpc: str, bsc_rpc: str):
        self.w3_eth = Web3(Web3.HTTPProvider(eth_rpc))
        self.w3_bsc = Web3(Web3.HTTPProvider(bsc_rpc))
        
        # Minimum whale thresholds (USD)
        self.whale_thresholds = {
            'BTC': 1000000,    # $1M
            'ETH': 500000,     # $500K
            'BNB': 250000,     # $250K
            'USDT': 5000000,   # $5M
            'USDC': 5000000,   # $5M
            'default': 100000  # $100K
        }
        
        # Known exchange addresses
        self.exchange_addresses = {
            # Binance
            '0x28C6c06298d514Db089934071355E5743bf21d60': 'Binance',
            '0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549': 'Binance',
            '0xDFd5293D8e347dFe59E90eFd55b2956a1343963d': 'Binance',
            
            # Coinbase
            '0x71660c4005BA85c37ccec55d0C4493E66Fe775d3': 'Coinbase',
            '0x503828976D22510aad0201ac7EC88293211D23Da': 'Coinbase',
            
            # Kraken
            '0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2': 'Kraken',
            '0x0A869d79a7052C7f1b55a8EbAbbEa3420F0D1E13': 'Kraken',
            
            # FTX (historical)
            '0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2': 'FTX',
            
            # Add more as needed
        }
        
        # Smart money addresses (known profitable wallets)
        self.smart_money_addresses = set()
        
    async def track_whale_movements(self, token_address: str, 
                                   network: str = 'ethereum') -> List[WhaleTransaction]:
        """Token için whale hareketlerini takip et"""
        try:
            w3 = self.w3_eth if network == 'ethereum' else self.w3_bsc
            
            # Get latest block
            latest_block = w3.eth.get_block('latest').number
            from_block = latest_block - 100  # Last 100 blocks
            
            # ERC20 Transfer event signature
            transfer_topic = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
            
            # Get transfer events
            logs = w3.eth.get_logs({
                'address': token_address,
                'fromBlock': from_block,
                'toBlock': latest_block,
                'topics': [transfer_topic]
            })
            
            whale_txs = []
            
            for log in logs:
                # Decode transfer event
                tx_data = self._decode_transfer_event(log, w3)
                
                if tx_data and self._is_whale_transaction(tx_data):
                    # Classify transaction type
                    tx_type = self._classify_transaction(
                        tx_data['from'],
                        tx_data['to']
                    )
                    
                    # Get USD value
                    usd_value = await self._get_usd_value(
                        tx_data['token'],
                        tx_data['amount']
                    )
                    
                    whale_tx = WhaleTransaction(
                        tx_hash=tx_data['tx_hash'],
                        from_address=tx_data['from'],
                        to_address=tx_data['to'],
                        token=tx_data['token'],
                        amount=tx_data['amount'],
                        value_usd=usd_value,
                        timestamp=datetime.now(),
                        tx_type=tx_type
                    )
                    
                    whale_txs.append(whale_tx)
            
            return whale_txs
            
        except Exception as e:
            logger.error(f"Whale tracking error: {e}")
            return []
    
    def _decode_transfer_event(self, log, w3) -> Optional[Dict]:
        """Transfer event'ini decode et"""
        try:
            # Topics: [event_signature, from, to]
            # Data: amount
            
            from_address = '0x' + log['topics'][1].hex()[26:]
            to_address = '0x' + log['topics'][2].hex()[26:]
            amount = int(log['data'].hex(), 16) / 1e18  # Assuming 18 decimals
            
            return {
                'tx_hash': log['transactionHash'].hex(),
                'from': from_address,
                'to': to_address,
                'token': log['address'],
                'amount': amount,
                'block': log['blockNumber']
            }
            
        except Exception as e:
            logger.error(f"Error decoding transfer: {e}")
            return None
    
    def _is_whale_transaction(self, tx_data: Dict) -> bool:
        """Whale transaction mı kontrol et"""
        threshold = self.whale_thresholds.get(
            tx_data.get('token_symbol', ''),
            self.whale_thresholds['default']
        )
        
        # Simple check based on amount
        # In production, would need to get token price
        return tx_data['amount'] > 1000  # Placeholder
    
    def _classify_transaction(self, from_addr: str, to_addr: str) -> str:
        """Transaction tipini belirle"""
        from_exchange = from_addr.lower() in [a.lower() for a in self.exchange_addresses]
        to_exchange = to_addr.lower() in [a.lower() for a in self.exchange_addresses]
        
        if from_exchange and not to_exchange:
            return 'exchange_withdrawal'
        elif to_exchange and not from_exchange:
            return 'exchange_deposit'
        elif from_exchange and to_exchange:
            return 'exchange_transfer'
        else:
            return 'wallet_transfer'
    
    async def _get_usd_value(self, token: str, amount: float) -> float:
        """USD değerini hesapla"""
        # CoinGecko API veya oracle kullan
        try:
            # Placeholder - gerçek implementasyonda API kullan
            prices = {
                'ETH': 2500,
                'BTC': 45000,
                'BNB': 300,
                'USDT': 1,
                'USDC': 1
            }
            
            return amount * prices.get(token, 0)
            
        except:
            return 0
    
    async def analyze_smart_money(self) -> Dict:
        """Smart money (karlı cüzdanlar) analizi"""
        analysis = {
            'accumulating_tokens': [],
            'distributing_tokens': [],
            'new_positions': [],
            'closed_positions': []
        }
        
        # Track known profitable wallets
        for address in self.smart_money_addresses:
            # Get recent transactions
            txs = await self._get_wallet_transactions(address)
            
            # Analyze patterns
            buying = [tx for tx in txs if tx.tx_type == 'buy']
            selling = [tx for tx in txs if tx.tx_type == 'sell']
            
            if len(buying) > len(selling):
                # Accumulation phase
                tokens = list(set([tx.token for tx in buying]))
                analysis['accumulating_tokens'].extend(tokens)
            elif len(selling) > len(buying):
                # Distribution phase
                tokens = list(set([tx.token for tx in selling]))
                analysis['distributing_tokens'].extend(tokens)
        
        return analysis
    
    async def _get_wallet_transactions(self, address: str) -> List[WhaleTransaction]:
        """Cüzdan işlemlerini al"""
        # Implementation would query blockchain
        return []
    
    async def detect_unusual_activity(self) -> List[Dict]:
        """Anormal aktiviteleri tespit et"""
        alerts = []
        
        # Check for unusual patterns
        # 1. Sudden large deposits to exchanges (potential dump)
        # 2. Large withdrawals from exchanges (potential pump)
        # 3. New whale wallets appearing
        # 4. Dormant wallets becoming active
        
        return alerts

# ============================================
# AGGREGATE SOCIAL SENTIMENT
# ============================================

class SocialSentimentAggregator:
    """Tüm sosyal platformlardan sentiment toplama"""
    
    def __init__(self):
        self.reddit_analyzer = None
        self.discord_analyzer = None
        self.twitter_analyzer = None  # To be implemented
        
        self.sentiment_history = defaultdict(list)
        self.alert_thresholds = {
            'extreme_positive': 0.7,
            'extreme_negative': -0.7,
            'volume_spike': 3.0  # 3x normal volume
        }
    
    async def aggregate_sentiment(self) -> Dict:
        """Tüm kaynaklardan sentiment topla"""
        aggregated = defaultdict(lambda: {
            'reddit_sentiment': 0,
            'discord_sentiment': 0,
            'twitter_sentiment': 0,
            'total_mentions': 0,
            'weighted_sentiment': 0,
            'trend': 'neutral',
            'signals': []
        })
        
        # Reddit sentiment
        if self.reddit_analyzer:
            for subreddit in ['cryptocurrency', 'bitcoin']:
                results = await self.reddit_analyzer.analyze_subreddit_sentiment(
                    subreddit, limit=50
                )
                for coin, data in results.items():
                    aggregated[coin]['reddit_sentiment'] = data['weighted_sentiment']
                    aggregated[coin]['total_mentions'] += data['mentions']
        
        # Discord sentiment
        if self.discord_analyzer:
            # Get from cache
            pass
        
        # Calculate weighted average
        for coin, data in aggregated.items():
            weights = {
                'reddit': 0.4,
                'discord': 0.3,
                'twitter': 0.3
            }
            
            weighted = (
                data['reddit_sentiment'] * weights['reddit'] +
                data['discord_sentiment'] * weights['discord'] +
                data['twitter_sentiment'] * weights['twitter']
            )
            
            data['weighted_sentiment'] = weighted
            
            # Generate signals
            if weighted > self.alert_thresholds['extreme_positive']:
                data['signals'].append({
                    'type': 'extreme_positive_sentiment',
                    'strength': weighted,
                    'action': 'consider_buy'
                })
            elif weighted < self.alert_thresholds['extreme_negative']:
                data['signals'].append({
                    'type': 'extreme_negative_sentiment',
                    'strength': weighted,
                    'action': 'consider_sell'
                })
            
            # Check volume spike
            if data['total_mentions'] > self._get_average_mentions(coin) * 3:
                data['signals'].append({
                    'type': 'volume_spike',
                    'mentions': data['total_mentions'],
                    'action': 'high_interest'
                })
        
        return dict(aggregated)
    
    def _get_average_mentions(self, coin: str) -> float:
        """Ortalama mention sayısını al"""
        if coin in self.sentiment_history and self.sentiment_history[coin]:
            return np.mean([h['mentions'] for h in self.sentiment_history[coin][-24:]])
        return 10  # Default value
    
    async def generate_report(self) -> str:
        """Sosyal sentiment raporu oluştur"""
        sentiment = await self.aggregate_sentiment()
        
        report = "📊 SOSYAL SENTIMENT RAPORU\n"
        report += "=" * 40 + "\n\n"
        
        # Top positive sentiment
        positive = sorted(
            sentiment.items(),
            key=lambda x: x[1]['weighted_sentiment'],
            reverse=True
        )[:5]
        
        report += "🟢 EN POZİTİF SENTIMENT:\n"
        for coin, data in positive:
            report += f"  {coin}: {data['weighted_sentiment']:.2f} "
            report += f"({data['total_mentions']} mention)\n"
        
        # Top negative sentiment
        negative = sorted(
            sentiment.items(),
            key=lambda x: x[1]['weighted_sentiment']
        )[:5]
        
        report += "\n🔴 EN NEGATİF SENTIMENT:\n"
        for coin, data in negative:
            report += f"  {coin}: {data['weighted_sentiment']:.2f} "
            report += f"({data['total_mentions']} mention)\n"
        
        # Trending coins
        report += "\n🔥 TREND OLAN COİNLER:\n"
        trending = [
            (coin, data) for coin, data in sentiment.items()
            if data['total_mentions'] > self._get_average_mentions(coin) * 2
        ]
        
        for coin, data in trending[:5]:
            report += f"  {coin}: {data['total_mentions']} mention "
            report += f"(sentiment: {data['weighted_sentiment']:.2f})\n"
        
        return report