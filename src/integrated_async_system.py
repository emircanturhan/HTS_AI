class IntegratedAsyncSystem:
    """Tüm modülleri entegre eden ana sistem"""
    
    def __init__(self):
        # Configuration
        self.config = AsyncConfig(
            max_concurrent_requests=500,
            max_workers=200,
            batch_size=100
        )
        
        # Components
        self.pipeline = AsyncPipelineOrchestrator(self.config)
        self.monitor = AsyncPerformanceMonitor()
        self.cache = AsyncRedisCache()
        self.dashboard = AsyncMonitoringDashboard(self.monitor)
        
        # Data sources
        self.sources = self._load_sources()
        
    def _load_sources(self) -> List[str]:
        """Veri kaynaklarını yükle"""
        
        return [
            # APIs
            'https://api.coindesk.com/v1/news',
            'https://cryptopanic.com/api/v1/posts/',
            'https://messari.io/api/v1/news',
            
            # RSS Feeds
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://cointelegraph.com/rss',
            
            # Web Scraping
            'https://www.reuters.com/technology/crypto',
            'https://www.bloomberg.com/crypto',
            
            # Social Media
            'https://api.twitter.com/2/tweets/search/recent',
            
            # Regulatory
            'https://www.sec.gov/rss/news/press.xml',
        ]
    
    async def start(self):
        """Sistemi başlat"""
        
        print("🚀 Starting Integrated Async System...")
        
        # Initialize components
        await self.cache.initialize()
        await self.pipeline.initialize()
        
        # Start dashboard
        dashboard_task = asyncio.create_task(
            self.dashboard.start(port=8080)
        )
        
        # Start monitoring
        monitor_task = asyncio.create_task(
            self.monitor.print_metrics()
        )
        
        # Start main pipeline
        pipeline_task = asyncio.create_task(
            self.pipeline.run(self.sources, interval=60)
        )
        
        # Wait for all tasks
        await asyncio.gather(
            dashboard_task,
            monitor_task,
            pipeline_task
        )
    
    async def shutdown(self):
        """Sistemi kapat"""
        
        print("Shutting down system...")
        
        await self.pipeline.shutdown()
        await self.cache.close()
        
        print("System shutdown complete")

# Ana çalıştırma
async def main():
    system = IntegratedAsyncSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    # Use uvloop for maximum performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Run with optimized event loop
    asyncio.run(main())