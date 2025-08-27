from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta
import psutil
import asyncio
from typing import Dict, List
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Prometheus metrikleri
api_calls_counter = Counter('api_calls_total', 'Total API calls', ['source'])
signal_accuracy_gauge = Gauge('signal_accuracy', 'Signal accuracy percentage', ['coin'])
processing_time_histogram = Histogram('processing_time_seconds', 'Processing time', ['operation'])
active_positions_gauge = Gauge('active_positions', 'Number of active positions')

class SystemMonitor:
    """Sistem monitoring sınıfı"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'api_calls': {},
            'signal_performance': {},
            'active_alerts': [],
            'error_logs': []
        }
        self.alert_thresholds = {
            'cpu_critical': 90,
            'memory_critical': 85,
            'api_rate_limit': 0.8,  # %80 rate limit kullanımı
            'signal_accuracy_min': 0.6,
            'max_drawdown': 0.15
        }
    
    def collect_system_metrics(self):
        """Sistem metriklerini topla"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'process_count': len(psutil.pids())
        }
        
        # Kritik durum kontrolü
        if metrics['cpu_percent'] > self.alert_thresholds['cpu_critical']:
            self.trigger_alert('CPU_CRITICAL', f"CPU usage: {metrics['cpu_percent']}%")
        
        if metrics['memory_percent'] > self.alert_thresholds['memory_critical']:
            self.trigger_alert('MEMORY_CRITICAL', f"Memory usage: {metrics['memory_percent']}%")
        
        return metrics
    
    def monitor_api_usage(self, source: str, used: int, limit: int):
        """API kullanımını takip et"""
        usage_ratio = used / limit if limit > 0 else 0
        
        if source not in self.metrics['api_calls']:
            self.metrics['api_calls'][source] = []
        
        self.metrics['api_calls'][source].append({
            'timestamp': datetime.now().isoformat(),
            'used': used,
            'limit': limit,
            'usage_ratio': usage_ratio
        })
        
        # Rate limit uyarısı
        if usage_ratio > self.alert_thresholds['api_rate_limit']:
            self.trigger_alert('API_RATE_LIMIT', 
                             f"{source} API usage: {usage_ratio:.1%} of limit")
        
        # Prometheus metric güncelle
        api_calls_counter.labels(source=source).inc()
    
    def track_signal_performance(self, coin: str, signal_id: str, outcome: Dict):
        """Sinyal performansını takip et"""
        if coin not in self.metrics['signal_performance']:
            self.metrics['signal_performance'][coin] = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'accuracy': 0.0,
                'profit_loss': 0.0
            }
        
        perf = self.metrics['signal_performance'][coin]
        perf['total'] += 1
        
        if outcome['success']:
            perf['successful'] += 1
        else:
            perf['failed'] += 1
        
        perf['accuracy'] = perf['successful'] / perf['total']
        perf['profit_loss'] += outcome.get('pnl', 0)
        
        # Prometheus metric güncelle
        signal_accuracy_gauge.labels(coin=coin).set(perf['accuracy'])
        
        # Düşük accuracy uyarısı
        if perf['accuracy'] < self.alert_thresholds['signal_accuracy_min'] and perf['total'] > 10:
            self.trigger_alert('LOW_ACCURACY', 
                             f"{coin} signal accuracy: {perf['accuracy']:.1%}")
    
    def trigger_alert(self, alert_type: str, message: str, severity: str = 'warning'):
        """Alert tetikle"""
        alert = {
            'id': f"{alert_type}_{datetime.now().timestamp()}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }
        
        self.metrics['active_alerts'].append(alert)
        
        # WebSocket ile anlık bildirim
        socketio.emit('new_alert', alert)
        
        # Kritik alertler için özel işlem
        if severity == 'critical':
            self._handle_critical_alert(alert)
        
        return alert
    
    def _handle_critical_alert(self, alert: Dict):
        """Kritik alertleri işle"""
        # Telegram bildirimi gönder
        # Email gönder
        # Otomatik düzeltme aksiyonları
        pass
    
    def get_health_status(self) -> Dict:
        """Sistem sağlık durumunu al"""
        system_metrics = self.collect_system_metrics()
        
        # Genel sağlık skoru hesapla
        health_score = 100
        
        # CPU bazlı düşüş
        if system_metrics['cpu_percent'] > 70:
            health_score -= (system_metrics['cpu_percent'] - 70) * 0.5
        
        # Memory bazlı düşüş
        if system_metrics['memory_percent'] > 70:
            health_score -= (system_metrics['memory_percent'] - 70) * 0.5
        
        # Active alert bazlı düşüş
        critical_alerts = [a for a in self.metrics['active_alerts'] 
                          if a['severity'] == 'critical' and not a['acknowledged']]
        health_score -= len(critical_alerts) * 10
        
        health_status = 'healthy' if health_score > 80 else 'degraded' if health_score > 60 else 'critical'
        
        return {
            'status': health_status,
            'score': max(0, min(100, health_score)),
            'metrics': system_metrics,
            'active_alerts': len(self.metrics['active_alerts']),
            'critical_alerts': len(critical_alerts)
        }

# ============================================
# WEB DASHBOARD ROUTES
# ============================================

monitor = SystemMonitor()

@app.route('/')
def dashboard():
    """Ana dashboard sayfası"""
    return render_template('dashboard.html')

@app.route('/api/health')
def health_check():
    """Sağlık durumu endpoint'i"""
    return jsonify(monitor.get_health_status())

@app.route('/api/metrics')
def get_metrics():
    """Metrikleri al"""
    return jsonify(monitor.metrics)

@app.route('/api/performance/<coin>')
def get_coin_performance(coin):
    """Coin bazlı performans"""
    if coin in monitor.metrics['signal_performance']:
        return jsonify(monitor.metrics['signal_performance'][coin])
    return jsonify({'error': 'Coin not found'}), 404

@app.route('/api/alerts', methods=['GET', 'POST'])
def handle_alerts():
    """Alert yönetimi"""
    if request.method == 'GET':
        return jsonify(monitor.metrics['active_alerts'])
    
    elif request.method == 'POST':
        action = request.json.get('action')
        alert_id = request.json.get('alert_id')
        
        if action == 'acknowledge':
            for alert in monitor.metrics['active_alerts']:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    return jsonify({'status': 'success'})
        
        return jsonify({'error': 'Alert not found'}), 404

@app.route('/api/charts/<chart_type>')
def get_chart_data(chart_type):
    """Grafik verilerini al"""
    if chart_type == 'cpu_usage':
        data = monitor.metrics.get('cpu_usage', [])[-100:]  # Son 100 veri
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in data],
            y=[d['value'] for d in data],
            mode='lines',
            name='CPU Usage'
        ))
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON
    
    return jsonify({'error': 'Chart type not found'}), 404

# ============================================
# WEBSOCKET EVENTS
# ============================================

@socketio.on('connect')
def handle_connect():
    """WebSocket bağlantısı"""
    print('Client connected')
    emit('connected', {'data': 'Connected to monitoring server'})

@socketio.on('subscribe_metrics')
def handle_subscribe(data):
    """Metrik aboneliği"""
    metric_type = data.get('type', 'all')
    
    # Periyodik güncelleme başlat
    socketio.start_background_task(send_periodic_updates, metric_type)

def send_periodic_updates(metric_type):
    """Periyodik metrik güncellemeleri gönder"""
    while True:
        socketio.sleep(5)  # 5 saniyede bir güncelle
        
        if metric_type == 'system' or metric_type == 'all':
            metrics = monitor.collect_system_metrics()
            socketio.emit('system_metrics', metrics)
        
        if metric_type == 'performance' or metric_type == 'all':
            perf_data = monitor.metrics['signal_performance']
            socketio.emit('performance_metrics', perf_data)

# ============================================
# ALERT AUTOMATION
# ============================================

class AlertAutomation:
    """Otomatik alert yönetimi ve düzeltme"""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.automation_rules = {
            'CPU_CRITICAL': self._handle_cpu_critical,
            'MEMORY_CRITICAL': self._handle_memory_critical,
            'API_RATE_LIMIT': self._handle_rate_limit,
            'LOW_ACCURACY': self._handle_low_accuracy
        }
    
    async def process_alert(self, alert: Dict):
        """Alert'i işle ve otomatik aksiyon al"""
        alert_type = alert['type']
        
        if alert_type in self.automation_rules:
            handler = self.automation_rules[alert_type]
            await handler(alert)
    
    async def _handle_cpu_critical(self, alert: Dict):
        """CPU kritik durumunu yönet"""
        # Düşük öncelikli işlemleri durdur
        # Cache temizle
        # Process sayısını azalt
        pass
    
    async def _handle_memory_critical(self, alert: Dict):
        """Memory kritik durumunu yönet"""
        # Unused cache'leri temizle
        # Büyük dataframe'leri disk'e yaz
        # Garbage collection tetikle
        import gc
        gc.collect()
    
    async def _handle_rate_limit(self, alert: Dict):
        """API rate limit durumunu yönet"""
        # API çağrılarını yavaşlat
        # Backup API'ye geç
        # Cache'den veri kullan
        pass
    
    async def _handle_low_accuracy(self, alert: Dict):
        """Düşük accuracy durumunu yönet"""
        # Model retraining tetikle
        # Daha conservative parametrelere geç
        # İşlem boyutunu küçült
        pass

# ============================================
# PERFORMANCE ANALYZER
# ============================================

class PerformanceAnalyzer:
    """Detaylı performans analizi"""
    
    def __init__(self):
        self.metrics_history = []
        
    def analyze_strategy_performance(self, strategy_name: str, 
                                    timeframe: str = '24h') -> Dict:
        """Strateji performansını analiz et"""
        
        # Zaman aralığını belirle
        if timeframe == '24h':
            start_time = datetime.now() - timedelta(hours=24)
        elif timeframe == '7d':
            start_time = datetime.now() - timedelta(days=7)
        elif timeframe == '30d':
            start_time = datetime.now() - timedelta(days=30)
        else:
            start_time = datetime.now() - timedelta(hours=24)
        
        # Metrikleri filtrele
        relevant_metrics = [m for m in self.metrics_history 
                          if m['timestamp'] > start_time and 
                          m['strategy'] == strategy_name]
        
        if not relevant_metrics:
            return {'error': 'No data available'}
        
        # Analiz yap
        total_trades = len(relevant_metrics)
        successful_trades = sum(1 for m in relevant_metrics if m['success'])
        total_pnl = sum(m['pnl'] for m in relevant_metrics)
        
        # Zaman bazlı performans
        hourly_performance = self._calculate_hourly_performance(relevant_metrics)
        
        # Coin bazlı performans
        coin_performance = self._calculate_coin_performance(relevant_metrics)
        
        return {
            'strategy': strategy_name,
            'timeframe': timeframe,
            'total_trades': total_trades,
            'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'best_hour': max(hourly_performance.items(), key=lambda x: x[1]),
            'worst_hour': min(hourly_performance.items(), key=lambda x: x[1]),
            'top_performing_coin': max(coin_performance.items(), key=lambda x: x[1]['pnl']),
            'worst_performing_coin': min(coin_performance.items(), key=lambda x: x[1]['pnl'])
        }
    
    def _calculate_hourly_performance(self, metrics: List[Dict]) -> Dict:
        """Saatlik performans hesapla"""
        hourly_pnl = {}
        
        for metric in metrics:
            hour = metric['timestamp'].hour
            if hour not in hourly_pnl:
                hourly_pnl[hour] = 0
            hourly_pnl[hour] += metric['pnl']
        
        return hourly_pnl
    
    def _calculate_coin_performance(self, metrics: List[Dict]) -> Dict:
        """Coin bazlı performans hesapla"""
        coin_stats = {}
        
        for metric in metrics:
            coin = metric['coin']
            if coin not in coin_stats:
                coin_stats[coin] = {'trades': 0, 'pnl': 0, 'success': 0}
            
            coin_stats[coin]['trades'] += 1
            coin_stats[coin]['pnl'] += metric['pnl']
            if metric['success']:
                coin_stats[coin]['success'] += 1
        
        # Success rate hesapla
        for coin in coin_stats:
            stats = coin_stats[coin]
            stats['success_rate'] = stats['success'] / stats['trades']
        
        return coin_stats

# ============================================
# SMART NOTIFICATIONS
# ============================================

class SmartNotificationSystem:
    """Akıllı bildirim sistemi"""
    
    def __init__(self):
        self.user_preferences = {
            'notification_channels': ['telegram', 'email'],
            'quiet_hours': {'start': 23, 'end': 7},
            'min_importance': 'medium',
            'categories': {
                'signals': True,
                'alerts': True,
                'performance': False,
                'news': True
            }
        }
        self.notification_queue = []
        
    def should_send_notification(self, notification: Dict) -> bool:
        """Bildirimin gönderilip gönderilmeyeceğini kontrol et"""
        
        # Kategori kontrolü
        category = notification.get('category', 'general')
        if category in self.user_preferences['categories']:
            if not self.user_preferences['categories'][category]:
                return False
        
        # Önem seviyesi kontrolü
        importance_levels = ['low', 'medium', 'high', 'critical']
        min_importance = self.user_preferences['min_importance']
        notification_importance = notification.get('importance', 'medium')
        
        if importance_levels.index(notification_importance) < importance_levels.index(min_importance):
            return False
        
        # Sessiz saat kontrolü (kritik değilse)
        if notification_importance != 'critical':
            current_hour = datetime.now().hour
            quiet_start = self.user_preferences['quiet_hours']['start']
            quiet_end = self.user_preferences['quiet_hours']['end']
            
            if quiet_start <= current_hour or current_hour < quiet_end:
                # Sessiz saatlerde, kuyruğa ekle
                self.notification_queue.append(notification)
                return False
        
        return True
    
    async def send_smart_notification(self, notification: Dict):
        """Akıllı bildirim gönder"""
        
        if not self.should_send_notification(notification):
            return
        
        # Bildirimi zenginleştir
        enriched = self._enrich_notification(notification)
        
        # Kanalları belirle
        channels = self._determine_channels(enriched)
        
        # Gönder
        for channel in channels:
            await self._send_via_channel(enriched, channel)
    
    def _enrich_notification(self, notification: Dict) -> Dict:
        """Bildirimi ek bilgilerle zenginleştir"""
        
        # Bağlam ekle
        if notification['category'] == 'signals':
            # İlgili coin'in son performansını ekle
            # Benzer geçmiş sinyallerin sonuçlarını ekle
            pass
        
        # Emoji ekle
        notification['emoji'] = self._get_emoji(notification)
        
        # Aksiyon önerileri ekle
        notification['suggested_actions'] = self._get_suggested_actions(notification)
        
        return notification
    
    def _determine_channels(self, notification: Dict) -> List[str]:
        """Bildirim kanallarını belirle"""
        
        importance = notification.get('importance', 'medium')
        
        if importance == 'critical':
            # Tüm kanallar
            return self.user_preferences['notification_channels']
        elif importance == 'high':
            # Öncelikli kanal
            return [self.user_preferences['notification_channels'][0]]
        else:
            # Default kanal
            return [self.user_preferences['notification_channels'][0]]
    
    def _get_emoji(self, notification: Dict) -> str:
        """Bildirim tipine göre emoji seç"""
        emojis = {
            'buy_signal': '🟢',
            'sell_signal': '🔴',
            'alert': '⚠️',
            'critical': '🚨',
            'performance': '📊',
            'news': '📰'
        }
        return emojis.get(notification.get('type'), '📌')
    
    def _get_suggested_actions(self, notification: Dict) -> List[str]:
        """Önerilen aksiyonları belirle"""
        actions = []
        
        if notification['category'] == 'signals':
            actions.extend([
                'Review position size',
                'Check correlated assets',
                'Set stop-loss orders'
            ])
        elif notification['category'] == 'alerts':
            actions.extend([
                'Check system status',
                'Review recent trades',
                'Adjust risk parameters'
            ])
        
        return actions
    
    async def _send_via_channel(self, notification: Dict, channel: str):
        """Belirli kanaldan bildirim gönder"""
        if channel == 'telegram':
            # Telegram bot ile gönder
            pass
        elif channel == 'email':
            # Email gönder
            pass
        elif channel == 'discord':
            # Discord webhook ile gönder
            pass

# ============================================
# MAIN MONITORING STARTUP
# ============================================

if __name__ == '__main__':
    # Prometheus metrics endpoint
    @app.route('/metrics')
    def metrics():
        return prometheus_client.generate_latest()
    
    # Start monitoring
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)