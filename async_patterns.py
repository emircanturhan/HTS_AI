class AsyncPatterns:
    """Gelişmiş asenkron tasarım pattern'leri"""
    
    # 1. Producer-Consumer Pattern
    class AsyncProducerConsumer:
        def __init__(self):
            self.queue = asyncio.Queue()
            
        async def producer(self, n: int):
            """Veri üretici"""
            for i in range(n):
                item = f"item_{i}"
                await self.queue.put(item)
                await asyncio.sleep(0.1)  # Simulate work
            
            # Signal completion
            await self.queue.put(None)
        
        async def consumer(self, name: str):
            """Veri tüketici"""
            while True:
                item = await self.queue.get()
                
                if item is None:
                    # Signal to stop
                    await self.queue.put(None)  # For other consumers
                    break
                
                # Process item
                print(f"{name} processing {item}")
                await asyncio.sleep(0.2)  # Simulate work
                
                self.queue.task_done()
        
        async def run(self):
            """Producer-Consumer çalıştır"""
            # Start producer
            producer_task = asyncio.create_task(self.producer(10))
            
            # Start multiple consumers
            consumers = [
                asyncio.create_task(self.consumer(f"Consumer-{i}"))
                for i in range(3)
            ]
            
            # Wait for all
            await producer_task
            await asyncio.gather(*consumers)
    
    # 2. Scatter-Gather Pattern
    class AsyncScatterGather:
        async def scatter_gather(self, data: List, process_func: Callable):
            """Veriyi dağıt, işle ve topla"""
            
            # Scatter: Create tasks for each data item
            tasks = [
                asyncio.create_task(process_func(item))
                for item in data
            ]
            
            # Gather: Collect results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [
                r for r in results 
                if not isinstance(r, Exception)
            ]
            
            return valid_results
    
    # 3. Circuit Breaker Pattern
    class AsyncCircuitBreaker:
        def __init__(self, failure_threshold: int = 5, timeout: float = 60):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failures = 0
            self.last_failure_time = None
            self.state = 'closed'  # closed, open, half-open
            
        async def call(self, func: Callable, *args, **kwargs):
            """Circuit breaker ile fonksiyon çağrısı"""
            
            # Check circuit state
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                # Make the call
                result = await func(*args, **kwargs)
                
                # Reset on success
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failures = 0
                
                return result
                
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = 'open'
                    print(f"Circuit breaker opened after {self.failures} failures")
                
                raise e
    
    # 4. Bulkhead Pattern
    class AsyncBulkhead:
        def __init__(self, max_concurrent: int = 10):
            self.semaphore = asyncio.Semaphore(max_concurrent)
            
        async def execute(self, func: Callable, *args, **kwargs):
            """Bulkhead pattern ile execution"""
            
            async with self.semaphore:
                return await func(*args, **kwargs)
    
    # 5. Async Context Manager
    class AsyncResourceManager:
        async def __aenter__(self):
            """Async context enter"""
            print("Acquiring resource...")
            await asyncio.sleep(0.1)
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Async context exit"""
            print("Releasing resource...")
            await asyncio.sleep(0.1)
        
        async def use_resource(self):
            """Resource kullanımı"""
            print("Using resource...")
            await asyncio.sleep(0.2)