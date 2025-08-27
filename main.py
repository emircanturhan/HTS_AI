import asyncio
from src.core.kripto_hts_ai import KriptoHTSAI
from src.trading.automated_trading_execution import AutomatedTradingExecutor
from src.risk.portfolio_risk_management import ModernPortfolioOptimizer

async def main():
    # Initialize components
    app = KriptoHTSAI()
    trader = AutomatedTradingExecutor(exchange_configs)
    optimizer = ModernPortfolioOptimizer()
    
    # Start all systems
    tasks = [
        app.start(),
        trader.manage_positions(),
        # Add more tasks
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())