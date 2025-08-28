from aiohttp import web
import aiohttp_cors
import json

class AsyncMonitoringDashboard:
    """Async monitoring web dashboard"""
    
    def __init__(self, monitor: AsyncPerformanceMonitor):
        self.monitor = monitor
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
        
    def setup_routes(self):
        """Web routes kurulumu"""
        
        self.app.router.add_get('/metrics', self.handle_metrics)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_static('/', path='static', name='static')
        
    def setup_cors(self):
        """CORS kurulumu"""
        
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def handle_metrics(self, request):
        """Metrics endpoint"""
        
        metrics = await self.monitor.get_metrics()
        return web.json_response(metrics)
    
    async def handle_health(self, request):
        """Health check endpoint"""
        
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    
    async def websocket_handler(self, request):
        """WebSocket for real-time updates"""
        
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            # Send metrics every second
            while True:
                metrics = await self.monitor.get_metrics()
                await ws.send_str(json.dumps(metrics))
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await ws.close()
        
        return ws
    
    async def start(self, host='0.0.0.0', port=8080):
        """Dashboard'u başlat"""
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        print(f"Dashboard running at http://{host}:{port}")