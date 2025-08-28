import asyncio
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Artık Python'un 'src' klasörünü göreceğini varsayarak,
# doğru dosya adlarından import yapıyoruz.
from Kripto_HTS_AI import KriptoHTSAI
from automated_trading_execution import AutomatedTradingExecutor
from portfolio_risk_management import ModernPortfolioOptimizer

async def main():
    # .env dosyasından okunan exchange konfigürasyonları
    exchange_configs = {
        'binance': {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'futures': True
        }
    }

    # Bileşenleri başlat
    app = KriptoHTSAI()
    trader = AutomatedTradingExecutor(exchange_configs)
    optimizer = ModernPortfolioOptimizer()
    
    # Tüm sistemleri başlat
    tasks = [
        app.start(),
        trader.manage_positions(),
        # Gelecekte eklenecek diğer görevler
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())