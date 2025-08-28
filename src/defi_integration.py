from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import aiohttp
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================
# DeFi PROTOKOL YAPILANDIRMASI
# ============================================

DEFI_CONFIG = {
    'ethereum': {
        'rpc_url': 'https://mainnet.infura.io/v3/YOUR_INFURA_KEY',
        'chain_id': 1,
        'uniswap_v3_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        'uniswap_v3_factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
        'aave_v3_pool': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
        'compound_comptroller': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B'
    },
    'polygon': {
        'rpc_url': 'https://polygon-rpc.com',
        'chain_id': 137,
        'quickswap_router': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
        'aave_v3_pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'
    },
    'bsc': {
        'rpc_url': 'https://bsc-dataseed.binance.org',
        'chain_id': 56,
        'pancakeswap_router': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
        'venus_comptroller': '0xfD36E2c2a6789Db23113685031d7F16329158384'
    }
}

# Token addresses (Ethereum Mainnet)
TOKEN_ADDRESSES = {
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
    'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
    'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
    'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
    'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9'
}

@dataclass
class DeFiPosition:
    """DeFi pozisyon veri yapısı"""
    protocol: str
    chain: str
    asset: str
    amount: Decimal
    value_usd: Decimal
    apy: float
    health_factor: Optional[float] = None
    liquidation_price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class LiquidityPool:
    """Likidite havuzu veri yapısı"""
    protocol: str
    pair: str
    token0: str
    token1: str
    reserve0: Decimal
    reserve1: Decimal
    total_value_locked: Decimal
    volume_24h: Decimal
    fee_tier: float
    apy: float

# ============================================
# UNISWAP V3 ENTEGRASYONU
# ============================================

class UniswapV3Integration:
    """Uniswap V3 protokol entegrasyonu"""
    
    def __init__(self, network: str = 'ethereum'):
        self.network = network
        self.config = DEFI_CONFIG[network]
        self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
        
        if network == 'polygon':
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Load ABIs
        self.router_abi = self._load_abi('uniswap_v3_router.json')
        self.factory_abi = self._load_abi('uniswap_v3_factory.json')
        self.pool_abi = self._load_abi('uniswap_v3_pool.json')
        
        # Contract instances
        self.router = self.w3.eth.contract(
            address=self.config['uniswap_v3_router'],
            abi=self.router_abi
        )
        self.factory = self.w3.eth.contract(
            address=self.config['uniswap_v3_factory'],
            abi=self.factory_abi
        )
    
    def _load_abi(self, filename: str) -> list:
        """ABI dosyasını yükle"""
        try:
            with open(f'abis/{filename}', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"ABI file not found: {filename}")
            return []
    
    async def get_pool_info(self, token0: str, token1: str, fee: int = 3000) -> LiquidityPool:
        """Havuz bilgilerini al"""
        try:
            # Pool adresini bul
            pool_address = self.factory.functions.getPool(
                token0, token1, fee
            ).call()
            
            if pool_address == '0x0000000000000000000000000000000000000000':
                raise ValueError(f"Pool not found for {token0}/{token1}")
            
            # Pool contract
            pool = self.w3.eth.contract(address=pool_address, abi=self.pool_abi)
            
            # Slot0 verilerini al (price, tick, etc.)
            slot0 = pool.functions.slot0().call()
            sqrt_price_x96 = slot0[0]
            tick = slot0[1]
            
            # Likidite bilgileri
            liquidity = pool.functions.liquidity().call()
            
            # Fiyat hesaplama
            price = (sqrt_price_x96 / (2**96)) ** 2
            
            # 24 saatlik hacim (Graph API veya event logs'dan alınmalı)
            volume_24h = await self._get_24h_volume(pool_address)
            
            # APY hesaplama (basitleştirilmiş)
            fee_percent = fee / 1_000_000
            estimated_apy = (volume_24h * fee_percent * 365) / liquidity if liquidity > 0 else 0
            
            return LiquidityPool(
                protocol='Uniswap V3',
                pair=f"{token0}/{token1}",
                token0=token0,
                token1=token1,
                reserve0=Decimal(0),  # V3'te reserve yerine liquidity var
                reserve1=Decimal(0),
                total_value_locked=Decimal(liquidity),
                volume_24h=Decimal(volume_24h),
                fee_tier=fee_percent,
                apy=estimated_apy
            )
            
        except Exception as e:
            logger.error(f"Error getting pool info: {e}")
            return None
    
    async def get_quote(self, token_in: str, token_out: str, 
                       amount_in: int, fee: int = 3000) -> Dict:
        """Swap kotasyonu al"""
        try:
            # Quoter contract kullan
            quoter_address = '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6'
            quoter_abi = self._load_abi('uniswap_v3_quoter.json')
            quoter = self.w3.eth.contract(address=quoter_address, abi=quoter_abi)
            
            # Quote al
            amount_out = quoter.functions.quoteExactInputSingle(
                token_in,
                token_out,
                fee,
                amount_in,
                0
            ).call()
            
            # Price impact hesapla
            price_impact = self._calculate_price_impact(amount_in, amount_out)
            
            return {
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out': amount_out,
                'price': amount_out / amount_in if amount_in > 0 else 0,
                'price_impact': price_impact,
                'fee': fee,
                'protocol': 'Uniswap V3'
            }
            
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None
    
    def _calculate_price_impact(self, amount_in: int, amount_out: int) -> float:
        """Fiyat etkisini hesapla"""
        # Basitleştirilmiş hesaplama
        # Gerçek implementasyon için pool reserves ve liquidity derinliği gerekli
        if amount_in > 1e18:  # Büyük işlem
            return 0.05  # %5 impact
        elif amount_in > 1e17:  # Orta işlem
            return 0.02  # %2 impact
        else:
            return 0.005  # %0.5 impact
    
    async def _get_24h_volume(self, pool_address: str) -> float:
        """24 saatlik hacmi al (The Graph API)"""
        # Graph API endpoint
        graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        
        query = """
        query($pool: String!) {
            pool(id: $pool) {
                volumeUSD
                volumeToken0
                volumeToken1
            }
        }
        """
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    graph_url,
                    json={'query': query, 'variables': {'pool': pool_address.lower()}}
                ) as response:
                    data = await response.json()
                    return float(data['data']['pool']['volumeUSD'])
            except:
                return 0.0

# ============================================
# AAVE V3 ENTEGRASYONU
# ============================================

class AaveV3Integration:
    """Aave V3 lending protocol entegrasyonu"""
    
    def __init__(self, network: str = 'ethereum'):
        self.network = network
        self.config = DEFI_CONFIG[network]
        self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
        
        # Contract instances
        self.pool_abi = self._load_abi('aave_v3_pool.json')
        self.pool = self.w3.eth.contract(
            address=self.config['aave_v3_pool'],
            abi=self.pool_abi
        )
        
        # Oracle contract for prices
        self.oracle_address = '0x54586bE62E3c3580375aE3723C145253060Ca0C2'
    
    def _load_abi(self, filename: str) -> list:
        """ABI dosyasını yükle"""
        try:
            with open(f'abis/{filename}', 'r') as f:
                return json.load(f)
        except:
            return []
    
    async def get_user_position(self, user_address: str) -> DeFiPosition:
        """Kullanıcı pozisyonunu al"""
        try:
            # getUserAccountData fonksiyonunu çağır
            account_data = self.pool.functions.getUserAccountData(user_address).call()
            
            total_collateral_base = account_data[0] / 1e8  # 8 decimals
            total_debt_base = account_data[1] / 1e8
            available_borrow_base = account_data[2] / 1e8
            current_liquidation_threshold = account_data[3] / 10000  # Percentage
            ltv = account_data[4] / 10000
            health_factor = account_data[5] / 1e18
            
            # Liquidation price hesapla
            if total_debt_base > 0:
                liquidation_price = (total_debt_base * 1.05) / total_collateral_base
            else:
                liquidation_price = 0
            
            return DeFiPosition(
                protocol='Aave V3',
                chain=self.network,
                asset='Multi-Asset',
                amount=Decimal(total_collateral_base),
                value_usd=Decimal(total_collateral_base),
                apy=self._calculate_net_apy(user_address),
                health_factor=health_factor,
                liquidation_price=liquidation_price
            )
            
        except Exception as e:
            logger.error(f"Error getting Aave position: {e}")
            return None
    
    async def get_reserve_data(self, asset: str) -> Dict:
        """Rezerv verilerini al"""
        try:
            reserve_data = self.pool.functions.getReserveData(asset).call()
            
            # Unpack reserve data
            configuration = reserve_data[0]
            liquidity_index = reserve_data[1] / 1e27
            current_liquidity_rate = reserve_data[2] / 1e27 * 100  # APY percentage
            variable_borrow_rate = reserve_data[3] / 1e27 * 100
            current_variable_borrow_index = reserve_data[4] / 1e27
            atoken_address = reserve_data[8]
            
            return {
                'asset': asset,
                'liquidity_rate_apy': current_liquidity_rate,
                'variable_borrow_apy': variable_borrow_rate,
                'liquidity_index': liquidity_index,
                'utilization_rate': self._calculate_utilization(asset),
                'atoken_address': atoken_address,
                'available_liquidity': self._get_available_liquidity(asset)
            }
            
        except Exception as e:
            logger.error(f"Error getting reserve data: {e}")
            return None
    
    def _calculate_net_apy(self, user_address: str) -> float:
        """Net APY hesapla (supply APY - borrow APY)"""
        # Basitleştirilmiş hesaplama
        # Gerçek implementasyon için detaylı pozisyon verisi gerekli
        return 5.5  # %5.5 örnek değer
    
    def _calculate_utilization(self, asset: str) -> float:
        """Kullanım oranını hesapla"""
        # Total borrowed / Total supplied
        return 0.75  # %75 örnek değer
    
    def _get_available_liquidity(self, asset: str) -> float:
        """Mevcut likiditeyi al"""
        # Token contract'tan balance çek
        return 1000000  # Örnek değer

# ============================================
# DEFİ AGGREGATOR
# ============================================

class DeFiAggregator:
    """Çoklu DeFi protokol aggregator"""
    
    def __init__(self):
        self.uniswap = UniswapV3Integration()
        self.aave = AaveV3Integration()
        self.protocols = {
            'uniswap': self.uniswap,
            'aave': self.aave
        }
        
        # 1inch API for best routing
        self.inch_api_url = "https://api.1inch.io/v5.0/1"
    
    async def find_best_swap_route(self, token_in: str, token_out: str, 
                                   amount: int) -> Dict:
        """En iyi swap rotasını bul"""
        quotes = []
        
        # Uniswap quote
        uni_quote = await self.uniswap.get_quote(token_in, token_out, amount)
        if uni_quote:
            quotes.append(uni_quote)
        
        # 1inch quote
        inch_quote = await self._get_1inch_quote(token_in, token_out, amount)
        if inch_quote:
            quotes.append(inch_quote)
        
        # En iyi quote'u seç
        if quotes:
            best_quote = max(quotes, key=lambda x: x['amount_out'])
            return best_quote
        
        return None
    
    async def _get_1inch_quote(self, token_in: str, token_out: str, 
                               amount: int) -> Dict:
        """1inch API'den quote al"""
        params = {
            'fromTokenAddress': token_in,
            'toTokenAddress': token_out,
            'amount': amount,
            'fromAddress': '0x0000000000000000000000000000000000000000',
            'slippage': 1
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.inch_api_url}/quote",
                    params=params
                ) as response:
                    data = await response.json()
                    return {
                        'token_in': token_in,
                        'token_out': token_out,
                        'amount_in': amount,
                        'amount_out': int(data['toTokenAmount']),
                        'price': float(data['toTokenAmount']) / amount,
                        'price_impact': float(data.get('estimatedPriceImpact', 0)),
                        'protocol': '1inch',
                        'protocols_used': data.get('protocols', [])
                    }
            except Exception as e:
                logger.error(f"1inch API error: {e}")
                return None
    
    async def get_yield_opportunities(self) -> List[Dict]:
        """En iyi yield farming fırsatlarını bul"""
        opportunities = []
        
        # Aave lending rates
        for token in ['USDC', 'USDT', 'DAI']:
            if token in TOKEN_ADDRESSES:
                reserve_data = await self.aave.get_reserve_data(TOKEN_ADDRESSES[token])
                if reserve_data:
                    opportunities.append({
                        'protocol': 'Aave V3',
                        'asset': token,
                        'type': 'lending',
                        'apy': reserve_data['liquidity_rate_apy'],
                        'risk_level': 'low',
                        'min_amount': 0,
                        'liquidity': reserve_data['available_liquidity']
                    })
        
        # Uniswap LP opportunities
        pairs = [
            ('WETH', 'USDC', 3000),
            ('WBTC', 'WETH', 3000),
            ('USDC', 'USDT', 100)
        ]
        
        for token0, token1, fee in pairs:
            if token0 in TOKEN_ADDRESSES and token1 in TOKEN_ADDRESSES:
                pool_info = await self.uniswap.get_pool_info(
                    TOKEN_ADDRESSES[token0],
                    TOKEN_ADDRESSES[token1],
                    fee
                )
                if pool_info:
                    opportunities.append({
                        'protocol': 'Uniswap V3',
                        'asset': f"{token0}/{token1}",
                        'type': 'liquidity_provision',
                        'apy': pool_info.apy,
                        'risk_level': 'medium' if fee == 3000 else 'low',
                        'fee_tier': pool_info.fee_tier,
                        'tvl': float(pool_info.total_value_locked)
                    })
        
        # Sort by APY
        opportunities.sort(key=lambda x: x['apy'], reverse=True)
        
        return opportunities
    
    async def monitor_liquidations(self, health_threshold: float = 1.2) -> List[Dict]:
        """Yaklaşan liquidation'ları takip et"""
        liquidation_risks = []
        
        # Aave positions at risk
        # Graph API veya event logs kullanarak risk altındaki pozisyonları bul
        
        return liquidation_risks

# ============================================
# ARBITRAGE FINDER
# ============================================

class DeFiArbitrageFinder:
    """DeFi protokolleri arası arbitraj fırsatları"""
    
    def __init__(self):
        self.aggregator = DeFiAggregator()
        self.min_profit_threshold = 50  # Minimum $50 kar
        self.max_slippage = 0.02  # %2 max slippage
        
    async def find_arbitrage_opportunities(self) -> List[Dict]:
        """Arbitraj fırsatlarını bul"""
        opportunities = []
        
        # Token pairs to check
        pairs = [
            ('WETH', 'USDC'),
            ('WBTC', 'WETH'),
            ('DAI', 'USDC'),
            ('LINK', 'WETH')
        ]
        
        for token_a, token_b in pairs:
            if token_a in TOKEN_ADDRESSES and token_b in TOKEN_ADDRESSES:
                # Check A -> B -> A arbitrage
                opportunity = await self._check_triangular_arbitrage(
                    TOKEN_ADDRESSES[token_a],
                    TOKEN_ADDRESSES[token_b]
                )
                
                if opportunity and opportunity['profit_usd'] > self.min_profit_threshold:
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _check_triangular_arbitrage(self, token_a: str, token_b: str) -> Optional[Dict]:
        """Üçgen arbitraj kontrolü"""
        amount_in = int(1e18)  # 1 token
        
        # A -> B (Protocol 1)
        quote_1 = await self.aggregator.uniswap.get_quote(token_a, token_b, amount_in)
        if not quote_1:
            return None
        
        # B -> A (Protocol 2 veya farklı havuz)
        quote_2 = await self.aggregator._get_1inch_quote(
            token_b, token_a, quote_1['amount_out']
        )
        if not quote_2:
            return None
        
        # Kar hesapla
        final_amount = quote_2['amount_out']
        profit = final_amount - amount_in
        
        if profit > 0:
            # USD değerini hesapla (oracle kullan)
            profit_usd = self._calculate_usd_value(token_a, profit)
            
            return {
                'type': 'triangular',
                'path': [token_a, token_b, token_a],
                'protocols': [quote_1['protocol'], quote_2['protocol']],
                'amount_in': amount_in,
                'amount_out': final_amount,
                'profit': profit,
                'profit_usd': profit_usd,
                'profit_percentage': (profit / amount_in) * 100,
                'gas_estimate': 300000,  # Estimated gas
                'timestamp': datetime.now()
            }
        
        return None
    
    def _calculate_usd_value(self, token: str, amount: int) -> float:
        """USD değerini hesapla"""
        # Oracle veya fiyat feed kullan
        prices = {
            TOKEN_ADDRESSES['WETH']: 2500,
            TOKEN_ADDRESSES['WBTC']: 45000,
            TOKEN_ADDRESSES['USDC']: 1,
            TOKEN_ADDRESSES['DAI']: 1
        }
        
        return prices.get(token, 0) * (amount / 1e18)

# ============================================
# FLASH LOAN EXECUTOR
# ============================================

class FlashLoanExecutor:
    """Flash loan ile arbitraj execution"""
    
    def __init__(self, private_key: str):
        self.private_key = private_key
        self.w3 = Web3(Web3.HTTPProvider(DEFI_CONFIG['ethereum']['rpc_url']))
        self.account = self.w3.eth.account.from_key(private_key)
        
    async def execute_flash_loan_arbitrage(self, opportunity: Dict) -> Dict:
        """Flash loan ile arbitraj işlemini gerçekleştir"""
        try:
            # Flash loan contract'ını hazırla
            flash_loan_contract = self._prepare_flash_loan_contract()
            
            # Transaction parametreleri
            tx_params = {
                'from': self.account.address,
                'gas': opportunity['gas_estimate'],
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            }
            
            # Flash loan çağrısı
            # NOT: Gerçek implementasyon için smart contract gerekli
            
            return {
                'status': 'success',
                'tx_hash': '0x...',
                'profit': opportunity['profit_usd'],
                'gas_used': opportunity['gas_estimate']
            }
            
        except Exception as e:
            logger.error(f"Flash loan execution error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _prepare_flash_loan_contract(self):
        """Flash loan contract'ını hazırla"""
        # Smart contract deployment gerekli
        pass