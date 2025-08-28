import asyncio
import aiohttp
import aiofiles
import aiodns
import aioredis
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from asyncio import Queue, Semaphore, gather, create_task, as_completed
import uvloop  # Daha hızlı event loop
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial, lru_cache
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import traceback
import signal
import sys

# Ultra hızlı event loop kullan
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Async logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# ASYNC CONFIGURATION
# ============================================

@dataclass
class AsyncConfig:
    """Asenkron sistem konfigürasyonu"""
    
    # Concurrency limits
    max_concurrent_requests: int = 100
    max_connections_per_host: int = 30
    max_workers: int = 50
    
    # Timeouts
    request_timeout: float = 30.0
    connect_timeout: float = 10.0
    total_timeout: float = 300.0
    
    # Queue sizes
    task_queue_size: int = 1000
    result_queue_size: int = 5000
    
    # Batch processing
    batch_size: int = 50
    batch_timeout: float = 5.0
    
    # Rate limiting
    rate_limit_per_second: Dict[str, int] = field(default_factory=lambda: {
        'default': 10,
        'api': 50,
        'scraping': 5,
        'social': 20
    })
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds

# ============================================
# ASYNC PERFORMANCE MONITOR
# ============================================

class AsyncPerformanceMonitor:
    """Asenkron performans monitörü"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'throughput': 0
        })
        self.start_time = time.time()
        self.task_durations = defaultdict(deque)
        self.max_history = 1000
    
    async def record_task(self, task_type: str, duration: float, success: bool = True):
        """Task metriğini kaydet"""
        
        metrics = self.metrics[task_type]
        metrics['total_tasks'] += 1
        
        if success:
            metrics['completed_tasks'] += 1
        else:
            metrics['failed_tasks'] += 1
        
        # Duration statistics
        self.task_durations[task_type].append(duration)
        if len(self.task_durations[task_type]) > self.max_history:
            self.task_durations[task_type].popleft()
        
        durations = list(self.task_durations[task_type])
        metrics['avg_duration'] = np.mean(durations)
        metrics['min_duration'] = min(durations)
        metrics['max_duration'] = max(durations)
        
        # Throughput
        elapsed = time.time() - self.start_time
        metrics['throughput'] = metrics['completed_tasks'] / elapsed if elapsed > 0 else 0
    
    async def get_metrics(self) -> Dict:
        """Metrikleri al"""
        
        total_metrics = {
            'uptime': time.time() - self.start_time,
            'total_tasks': sum(m['total_tasks'] for m in self.metrics.values()),
            'completed_tasks': sum(m['completed_tasks'] for m in self.metrics.values()),
            'failed_tasks': sum(m['failed_tasks'] for m in self.metrics.values()),
            'task_types': dict(self.metrics)
        }
        
        # Success rate
        if total_metrics['total_tasks'] > 0:
            total_metrics['success_rate'] = total_metrics['completed_tasks'] / total_metrics['total_tasks']
        else:
            total_metrics['success_rate'] = 0
        
        return total_metrics
    
    async def print_metrics(self):
        """Metrikleri yazdır"""
        
        while True:
            await asyncio.sleep(60)  # Her dakika
            
            metrics = await self.get_metrics()
            logger.info(f"""
            ===== PERFORMANCE METRICS =====
            Uptime: {metrics['uptime']:.2f}s
            Total Tasks: {metrics['total_tasks']}
            Completed: {metrics['completed_tasks']}
            Failed: {metrics['failed_tasks']}
            Success Rate: {metrics['success_rate']:.2%}
            ================================
            """)
            
            for task_type, task_metrics in metrics['task_types'].items():
                logger.info(f"""
                {task_type}:
                  Throughput: {task_metrics['throughput']:.2f} tasks/s
                  Avg Duration: {task_metrics['avg_duration']:.3f}s
                  Min/Max: {task_metrics['min_duration']:.3f}s / {task_metrics['max_duration']:.3f}s
                """)

# ============================================
# ASYNC RATE LIMITER
# ============================================

class AsyncRateLimiter:
    """Asenkron rate limiter"""
    
    def __init__(self, rate: int, per: float = 1.0):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Rate limit kontrolü"""
        
        async with self.lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

# ============================================
# ASYNC CONNECTION POOL
# ============================================

class AsyncConnectionPool:
    """Optimized async connection pool"""
    
    def __init__(self, config: AsyncConfig):
        self.config = config
        self.sessions: Dict[str, ClientSession] = {}
        self.semaphores: Dict[str, Semaphore] = {}
        self.rate_limiters: Dict[str, AsyncRateLimiter] = {}
        
    async def initialize(self):
        """Connection pool'u başlat"""
        
        # Create connector with DNS caching
        connector = TCPConnector(
            limit=self.config.max_concurrent_requests,
            limit_per_host=self.config.max_connections_per_host,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True,
            keepalive_timeout=30
        )
        
        # Create timeout
        timeout = ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connect_timeout,
            sock_connect=self.config.connect_timeout,
            sock_read=self.config.request_timeout
        )
        
        # Create main session
        self.sessions['main'] = ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True
        )
        
        # Create specialized sessions
        for source_type in ['api', 'scraping', 'social']:
            self.sessions[source_type] = ClientSession(
                connector=TCPConnector(
                    limit=self.config.max_connections_per_host,
                    limit_per_host=10
                ),
                timeout=timeout
            )
            
            # Create semaphores for concurrency control
            self.semaphores[source_type] = Semaphore(
                self.config.rate_limit_per_second.get(source_type, 10)
            )
            
            # Create rate limiters
            self.rate_limiters[source_type] = AsyncRateLimiter(
                self.config.rate_limit_per_second.get(source_type, 10)
            )
    
    async def get_session(self, source_type: str = 'main') -> ClientSession:
        """Session al"""
        return self.sessions.get(source_type, self.sessions['main'])
    
    async def close(self):
        """Tüm session'ları kapat"""
        for session in self.sessions.values():
            await session.close()

# ============================================
# ASYNC DATA FETCHER
# ============================================

class AsyncDataFetcher:
    """Tam asenkron veri çekici"""
    
    def __init__(self, config: AsyncConfig):
        self.config = config
        self.connection_pool = AsyncConnectionPool(config)
        self.monitor = AsyncPerformanceMonitor()
        self.task_queue = Queue(maxsize=config.task_queue_size)
        self.result_queue = Queue(maxsize=config.result_queue_size)
        self.workers = []
        
    async def initialize(self):
        """Fetcher'ı başlat"""
        
        await self.connection_pool.initialize()
        
        # Start worker tasks
        for i in range(self.config.max_workers):
            worker = create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start monitor
        create_task(self.monitor.print_metrics())
        
        logger.info(f"AsyncDataFetcher initialized with {self.config.max_workers} workers")
    
    async def _worker(self, worker_id: str):
        """Worker task"""
        
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                # Process task
                start_time = time.time()
                
                try:
                    result = await self._process_task(task)
                    await self.result_queue.put(result)
                    
                    # Record metrics
                    duration = time.time() - start_time
                    await self.monitor.record_task(task['type'], duration, True)
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    
                    # Record failure
                    duration = time.time() - start_time
                    await self.monitor.record_task(task['type'], duration, False)
                    
                finally:
                    self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} critical error: {e}")
    
    async def _process_task(self, task: Dict) -> Dict:
        """Task'ı işle"""
        
        task_type = task['type']
        
        if task_type == 'fetch_url':
            return await self._fetch_url(task['url'], task.get('source_type', 'main'))
        elif task_type == 'fetch_api':
            return await self._fetch_api(task['url'], task.get('params', {}))
        elif task_type == 'parse_content':
            return await self._parse_content(task['content'], task.get('parser_type'))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _fetch_url(self, url: str, source_type: str) -> Dict:
        """URL'den veri çek"""
        
        # Get rate limiter
        rate_limiter = self.connection_pool.rate_limiters.get(source_type)
        if rate_limiter:
            await rate_limiter.acquire()
        
        # Get session
        session = await self.connection_pool.get_session(source_type)
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            'url': url,
                            'content': content,
                            'status': response.status,
                            'headers': dict(response.headers)
                        }
                    else:
                        if attempt < self.config.max_retries - 1:
                            delay = self.config.retry_delay * (2 ** attempt if self.config.exponential_backoff else 1)
                            await asyncio.sleep(delay)
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                            
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt if self.config.exponential_backoff else 1)
                    await asyncio.sleep(delay)
                else:
                    raise
        
        return {'url': url, 'error': 'Max retries exceeded'}
    
    async def fetch_multiple(self, urls: List[str]) -> List[Dict]:
        """Birden fazla URL'den paralel veri çek"""
        
        # Add tasks to queue
        for url in urls:
            await self.task_queue.put({
                'type': 'fetch_url',
                'url': url,
                'source_type': self._determine_source_type(url)
            })
        
        # Collect results
        results = []
        for _ in range(len(urls)):
            result = await self.result_queue.get()
            results.append(result)
        
        return results
    
    def _determine_source_type(self, url: str) -> str:
        """URL'den kaynak tipini belirle"""
        
        if 'api' in url:
            return 'api'
        elif any(social in url for social in ['twitter', 'reddit', 'facebook']):
            return 'social'
        else:
            return 'scraping'
    
    async def shutdown(self):
        """Fetcher'ı kapat"""
        
        # Stop workers
        for _ in self.workers:
            await self.task_queue.put(None)
        
        # Wait for workers to finish
        await gather(*self.workers)
        
        # Close connection pool
        await self.connection_pool.close()

# ============================================
# ASYNC NEWS PROCESSOR
# ============================================

class AsyncNewsProcessor:
    """Asenkron haber işleyici"""
    
    def __init__(self, config: AsyncConfig):
        self.config = config
        self.fetcher = AsyncDataFetcher(config)
        self.ai_analyzer = None  # Will be initialized
        self.processing_semaphore = Semaphore(config.max_workers)
        self.batch_processor = BatchProcessor(config.batch_size)
    
    async def initialize(self):
        """Processor'ı başlat"""
        
        await self.fetcher.initialize()
        
        # Initialize AI analyzer asynchronously
        self.ai_analyzer = await self._initialize_ai_analyzer()
        
        logger.info("AsyncNewsProcessor initialized")
    
    async def _initialize_ai_analyzer(self):
        """AI analyzer'ı asenkron başlat"""
        
        # Load models in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        
        # Import heavy modules in thread
        await loop.run_in_executor(executor, self._load_ai_modules)
        
        return self.ai_analyzer_instance
    
    def _load_ai_modules(self):
        """AI modüllerini yükle (blocking operation)"""
        
        # This runs in thread pool
        from advanced_ai_analyzer import AdvancedAIAnalyzer
        self.ai_analyzer_instance = AdvancedAIAnalyzer()
    
    async def process_news_stream(self, sources: List[str]) -> List[Dict]:
        """Haber akışını işle"""
        
        # Fetch news from all sources concurrently
        fetch_tasks = []
        
        for source in sources:
            task = create_task(self._fetch_from_source(source))
            fetch_tasks.append(task)
        
        # Wait for all fetches to complete
        fetch_results = await gather(*fetch_tasks, return_exceptions=True)
        
        # Flatten results
        all_articles = []
        for result in fetch_results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Fetch error: {result}")
        
        # Process articles in batches
        processed_articles = await self._process_articles_batch(all_articles)
        
        return processed_articles
    
    async def _fetch_from_source(self, source: str) -> List[Dict]:
        """Tek bir kaynaktan veri çek"""
        
        async with self.processing_semaphore:
            start_time = time.time()
            
            try:
                # Determine fetch method
                if source.startswith('http'):
                    result = await self.fetcher._fetch_url(source, 'scraping')
                    articles = await self._parse_articles(result['content'])
                else:
                    # API call
                    articles = await self._fetch_api_source(source)
                
                duration = time.time() - start_time
                logger.info(f"Fetched {len(articles)} articles from {source} in {duration:.2f}s")
                
                return articles
                
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                return []
    
    async def _process_articles_batch(self, articles: List[Dict]) -> List[Dict]:
        """Makaleleri batch halinde işle"""
        
        processed = []
        
        # Process in batches for efficiency
        batches = self.batch_processor.create_batches(articles)
        
        for batch in batches:
            # Process batch concurrently
            batch_tasks = []
            
            for article in batch:
                task = create_task(self._process_single_article(article))
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict):
                    processed.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Processing error: {result}")
        
        return processed
    
    async def _process_single_article(self, article: Dict) -> Dict:
        """Tek bir makaleyi işle"""
        
        async with self.processing_semaphore:
            try:
                # AI analysis
                if self.ai_analyzer:
                    analysis = await self._run_ai_analysis(article)
                    article['ai_analysis'] = analysis
                
                # Additional processing
                article['processed_at'] = datetime.now()
                article['processing_time'] = time.time()
                
                return article
                
            except Exception as e:
                logger.error(f"Article processing error: {e}")
                article['error'] = str(e)
                return article
    
    async def _run_ai_analysis(self, article: Dict) -> Dict:
        """AI analizi çalıştır"""
        
        # Run AI analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        
        result = await loop.run_in_executor(
            executor,
            self.ai_analyzer.analyze_comprehensive,
            article.get('content', ''),
            article.get('metadata', {})
        )
        
        return result

# ============================================
# BATCH PROCESSOR
# ============================================

class BatchProcessor:
    """Batch işleme optimizasyonu"""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    def create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Item'ları batch'lere böl"""
        
        batches = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    async def process_batches_async(self, 
                                   items: List[Any],
                                   process_func: Callable,
                                   max_concurrent: int = 10) -> List[Any]:
        """Batch'leri asenkron işle"""
        
        batches = self.create_batches(items)
        semaphore = Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await process_func(batch)
        
        tasks = [create_task(process_batch(batch)) for batch in batches]
        results = await gather(*tasks)
        
        # Flatten results
        flattened = []
        for batch_result in results:
            flattened.extend(batch_result)
        
        return flattened

# ============================================
# ASYNC PIPELINE ORCHESTRATOR
# ============================================

class AsyncPipelineOrchestrator:
    """Ana asenkron pipeline yöneticisi"""
    
    def __init__(self, config: AsyncConfig = None):
        self.config = config or AsyncConfig()
        self.processor = AsyncNewsProcessor(self.config)
        self.monitor = AsyncPerformanceMonitor()
        self.is_running = False
        self.tasks = []
        
    async def initialize(self):
        """Pipeline'ı başlat"""
        
        await self.processor.initialize()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AsyncPipelineOrchestrator initialized")
    
    def _signal_handler(self, sig, frame):
        """Graceful shutdown"""
        
        logger.info(f"Received signal {sig}, shutting down...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
    
    async def run(self, sources: List[str], interval: int = 60):
        """Ana pipeline'ı çalıştır"""
        
        self.is_running = True
        logger.info(f"Starting pipeline with {len(sources)} sources, interval={interval}s")
        
        while self.is_running:
            try:
                # Process news cycle
                cycle_start = time.time()
                
                # Create processing task
                process_task = create_task(self.processor.process_news_stream(sources))
                self.tasks.append(process_task)
                
                # Wait for processing or timeout
                try:
                    results = await asyncio.wait_for(
                        process_task,
                        timeout=self.config.total_timeout
                    )
                    
                    cycle_duration = time.time() - cycle_start
                    logger.info(f"Processed {len(results)} articles in {cycle_duration:.2f}s")
                    
                    # Handle results
                    await self._handle_results(results)
                    
                except asyncio.TimeoutError:
                    logger.error("Processing timeout exceeded")
                    process_task.cancel()
                
                # Wait for next cycle
                await asyncio.sleep(max(0, interval - (time.time() - cycle_start)))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(10)  # Error recovery delay
        
        # Cleanup
        await self.shutdown()
    
    async def _handle_results(self, results: List[Dict]):
        """Sonuçları işle"""
        
        # Store in database
        await self._store_results(results)
        
        # Send alerts if needed
        await self._check_alerts(results)
        
        # Update metrics
        metrics = await self.monitor.get_metrics()
        logger.info(f"Current throughput: {metrics.get('throughput', 0):.2f} items/s")
    
    async def _store_results(self, results: List[Dict]):
        """Sonuçları veritabanına kaydet"""
        
        # Async database operations
        # Implementation depends on database choice
        pass
    
    async def _check_alerts(self, results: List[Dict]):
        """Alert kontrolü"""
        
        # Check for high-impact news
        for result in results:
            if result.get('ai_analysis', {}).get('impact', {}).get('magnitude', 0) > 0.7:
                logger.warning(f"HIGH IMPACT: {result.get('title', 'Unknown')}")
                # Send alert
    
    async def shutdown(self):
        """Pipeline'ı kapat"""
        
        logger.info("Shutting down pipeline...")
        
        # Cancel remaining tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        await gather(*self.tasks, return_exceptions=True)
        
        # Shutdown processor
        await self.processor.fetcher.shutdown()
        
        logger.info("Pipeline shutdown complete")

# ============================================
# ASYNC REDIS CACHE
# ============================================

class AsyncRedisCache:
    """Asenkron Redis cache"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = None
        self.host = host
        self.port = port
        self.db = db
    
    async def initialize(self):
        """Redis bağlantısını başlat"""
        
        self.redis = await aioredis.create_redis_pool(
            f'redis://{self.host}:{self.port}/{self.db}',
            minsize=5,
            maxsize=10
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Cache'den al"""
        
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Cache'e kaydet"""
        
        if not self.redis:
            return
        
        try:
            await self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def close(self):
        """Redis bağlantısını kapat"""
        
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()

# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Ana çalıştırma fonksiyonu"""
    
    # Configuration
    config = AsyncConfig(
        max_concurrent_requests=200,
        max_workers=100,
        batch_size=100,
        rate_limit_per_second={
            'default': 20,
            'api': 100,
            'scraping': 10,
            'social': 50
        }
    )
    
    # News sources
    sources = [
        'https://api.coindesk.com/v1/news',
        'https://cryptopanic.com/api/v1/posts/',
        'https://newsapi.org/v2/everything?q=crypto',
        'https://www.reuters.com/technology/crypto',
        'https://www.bloomberg.com/crypto',
        # Add more sources
    ]
    
    # Initialize and run pipeline
    pipeline = AsyncPipelineOrchestrator(config)
    
    try:
        await pipeline.initialize()
        await pipeline.run(sources, interval=300)  # 5 minutes
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        await pipeline.shutdown()

if __name__ == "__main__":
    # Use uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main())