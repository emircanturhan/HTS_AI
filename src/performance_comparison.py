import asyncio
import time
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor

class PerformanceComparison:
    """Performans karşılaştırma testi"""
    
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.results = {}
    
    def test_synchronous(self):
        """Senkron test"""
        start = time.time()
        results = []
        
        for url in self.urls:
            try:
                response = requests.get(url, timeout=30)
                results.append(response.status_code)
            except:
                results.append(None)
        
        duration = time.time() - start
        self.results['synchronous'] = {
            'duration': duration,
            'throughput': len(self.urls) / duration,
            'success': len([r for r in results if r == 200])
        }
        
        return duration
    
    async def test_asynchronous(self):
        """Asenkron test"""
        start = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in self.urls:
                tasks.append(self._fetch_async(session, url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start
        self.results['asynchronous'] = {
            'duration': duration,
            'throughput': len(self.urls) / duration,
            'success': len([r for r in results if isinstance(r, int) and r == 200])
        }
        
        return duration
    
    async def _fetch_async(self, session, url):
        try:
            async with session.get(url) as response:
                return response.status
        except:
            return None
    
    def print_comparison(self):
        """Karşılaştırma sonuçlarını yazdır"""
        
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*50)
        
        sync_res = self.results.get('synchronous', {})
        async_res = self.results.get('asynchronous', {})
        
        print(f"\nSYNCHRONOUS:")
        print(f"  Duration: {sync_res.get('duration', 0):.2f}s")
        print(f"  Throughput: {sync_res.get('throughput', 0):.2f} req/s")
        print(f"  Success: {sync_res.get('success', 0)}/{len(self.urls)}")
        
        print(f"\nASYNCHRONOUS:")
        print(f"  Duration: {async_res.get('duration', 0):.2f}s")
        print(f"  Throughput: {async_res.get('throughput', 0):.2f} req/s")
        print(f"  Success: {async_res.get('success', 0)}/{len(self.urls)}")
        
        if sync_res and async_res:
            speedup = sync_res['duration'] / async_res['duration']
            print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster with async!")

# Test çalıştırma
async def run_performance_test():
    urls = [
        'https://api.coindesk.com/v1/news',
        'https://cryptopanic.com/api/v1/posts/',
        'https://newsapi.org/v2/everything',
        # ... 100+ URLs
    ] * 10  # 1000 requests
    
    tester = PerformanceComparison(urls)
    
    # Synchronous test
    print("Testing synchronous...")
    sync_time = tester.test_synchronous()
    
    # Asynchronous test
    print("Testing asynchronous...")
    async_time = await tester.test_asynchronous()
    
    # Results
    tester.print_comparison()

# Çalıştır
# asyncio.run(run_performance_test())