class AdvancedDataPipeline:
    """Gelişmiş veri işleme pipeline'ı"""
    
    def __init__(self):
        self.fetcher = EnhancedDataFetcher()
        self.ai_analyzer = AdvancedAIAnalyzer()
        self.impact_calculator = ImpactCalculator()
        self.alert_manager = AlertManager()
        self.cache = RedisCache()
        
    async def process_real_time(self):
        """Gerçek zamanlı veri işleme"""
        
        while True:
            try:
                # 1. Fetch from all sources
                articles = await self.fetcher.fetch_all()
                
                # 2. Filter duplicates and low quality
                filtered = self._quality_filter(articles)
                
                # 3. AI Analysis for each article
                analyzed = []
                for article in filtered:
                    # Check cache first
                    cached_analysis = await self.cache.get(article.hash_id)
                    
                    if cached_analysis:
                        analyzed.append(cached_analysis)
                    else:
                        # Perform AI analysis
                        analysis = await self.ai_analyzer.analyze_comprehensive(
                            article.content,
                            {
                                'title': article.title,
                                'source': article.source,
                                'url': article.url,
                                'published': article.published_date
                            }
                        )
                        
                        # Cache the analysis
                        await self.cache.set(article.hash_id, analysis, ttl=3600)
                        analyzed.append(analysis)
                
                # 4. Calculate aggregate market impact
                market_impact = self.impact_calculator.calculate_aggregate(analyzed)
                
                # 5. Generate alerts if needed
                alerts = await self.alert_manager.check_alerts(analyzed, market_impact)
                
                # 6. Update dashboard
                await self._update_dashboard(analyzed, market_impact, alerts)
                
                # 7. Store in database
                await self._store_results(analyzed, market_impact)
                
                logger.info(f"Processed {len(analyzed)} articles, {len(alerts)} alerts generated")
                
                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                await asyncio.sleep(30)
    
    def _quality_filter(self, articles: List[FetchedArticle]) -> List[FetchedArticle]:
        """Kalite filtreleme"""
        
        filtered = []
        
        for article in articles:
            # Minimum content length
            if len(article.content) < 50:
                continue
            
            # Check if it's actual news (not ads, etc.)
            if self._is_advertisement(article):
                continue
            
            # Language check
            if article.language not in ['en', 'tr']:
                continue
            
            # Source credibility
            if self._get_source_credibility(article.source) < 0.3:
                continue
            
            filtered.append(article)
        
        return filtered
    
    def _is_advertisement(self, article: FetchedArticle) -> bool:
        """Reklam içeriği kontrolü"""
        
        ad_keywords = [
            'sponsored', 'advertisement', 'promoted',
            'partner content', 'paid post', 'affiliate'
        ]
        
        content_lower = (article.title + ' ' + article.content).lower()
        return any(keyword in content_lower for keyword in ad_keywords)
    
    def _get_source_credibility(self, source: str) -> float:
        """Kaynak güvenilirlik skoru"""
        
        credibility_scores = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'coindesk': 0.85,
            'cointelegraph': 0.75,
            'twitter': 0.5,
            'reddit': 0.4
        }
        
        source_lower = source.lower()
        for key, score in credibility_scores.items():
            if key in source_lower:
                return score
        
        return 0.5  # Default

class ImpactCalculator:
    """Toplam piyasa etkisi hesaplayıcı"""
    
    def calculate_aggregate(self, analyses: List) -> Dict:
        """Toplam etki hesapla"""
        
        impact = {
            'overall_score': 0,
            'direction': 'neutral',
            'confidence': 0,
            'major_events': [],
            'affected_coins': {},
            'risk_level': 'low'
        }
        
        if not analyses:
            return impact
        
        # Weight by importance and credibility
        total_weight = 0
        weighted_sentiment = 0
        weighted_magnitude = 0
        
        for analysis in analyses:
            weight = analysis.impact.magnitude * analysis.credibility_score
            total_weight += weight
            
            weighted_sentiment += analysis.sentiment['compound'] * weight
            weighted_magnitude += analysis.impact.magnitude * weight
            
            # Track major events
            if analysis.impact.magnitude > 0.7:
                impact['major_events'].append({
                    'title': analysis.title,
                    'type': analysis.impact.event_type,
                    'magnitude': analysis.impact.magnitude
                })
            
            # Track affected coins
            for coin in analysis.impact.affected_assets:
                if coin not in impact['affected_coins']:
                    impact['affected_coins'][coin] = {
                        'mentions': 0,
                        'sentiment': 0,
                        'impact': 0
                    }
                
                impact['affected_coins'][coin]['mentions'] += 1
                impact['affected_coins'][coin]['sentiment'] += analysis.sentiment['compound']
                impact['affected_coins'][coin]['impact'] += analysis.impact.magnitude
        
        # Calculate overall metrics
        if total_weight > 0:
            impact['overall_score'] = weighted_magnitude / total_weight
            
            avg_sentiment = weighted_sentiment / total_weight
            if avg_sentiment > 0.2:
                impact['direction'] = 'bullish'
            elif avg_sentiment < -0.2:
                impact['direction'] = 'bearish'
            
            impact['confidence'] = min(total_weight / len(analyses), 1.0)
        
        # Determine risk level
        if impact['overall_score'] > 0.7:
            impact['risk_level'] = 'high'
        elif impact['overall_score'] > 0.4:
            impact['risk_level'] = 'medium'
        
        return impact

class AlertManager:
    """Akıllı alert yönetimi"""
    
    def __init__(self):
        self.alert_rules = self._load_alert_rules()
        self.alert_history = []
        self.notification_channels = ['telegram', 'email', 'discord']
    
    def _load_alert_rules(self) -> List[Dict]:
        """Alert kurallarını yükle"""
        
        return [
            {
                'name': 'Major Regulatory News',
                'condition': lambda a: a.impact.event_type == 'regulation' and a.impact.magnitude > 0.6,
                'priority': 'high',
                'message_template': "🚨 REGULATORY ALERT: {title}\nImpact: {magnitude:.2f}\nDirection: {direction}"
            },
            {
                'name': 'Security Breach',
                'condition': lambda a: a.impact.event_type == 'hack' and a.impact.magnitude > 0.5,
                'priority': 'critical',
                'message_template': "🔴 SECURITY ALERT: {title}\nAffected: {affected_assets}\nMagnitude: {magnitude:.2f}"
            },
            {
                'name': 'Major Partnership',
                'condition': lambda a: a.impact.event_type == 'partnership' and a.impact.magnitude > 0.7,
                'priority': 'medium',
                'message_template': "✅ PARTNERSHIP NEWS: {title}\nImpact: Positive\nConfidence: {confidence:.2%}"
            },
            {
                'name': 'Extreme Sentiment',
                'condition': lambda a: abs(a.sentiment.get('compound', 0)) > 0.8,
                'priority': 'medium',
                'message_template': "📊 SENTIMENT ALERT: {title}\nSentiment: {sentiment:.2f}\nSource: {source}"
            },
            {
                'name': 'Whale Movement',
                'condition': lambda a: 'whale' in a.title.lower() or 'large transfer' in a.title.lower(),
                'priority': 'medium',
                'message_template': "🐋 WHALE ALERT: {title}\nDetails: {content[:200]}..."
            }
        ]
    
    async def check_alerts(self, analyses: List, market_impact: Dict) -> List[Dict]:
        """Alert kontrolü yap"""
        
        alerts = []
        
        for analysis in analyses:
            for rule in self.alert_rules:
                try:
                    if rule['condition'](analysis):
                        alert = self._create_alert(analysis, rule, market_impact)
                        
                        # Check if not duplicate
                        if not self._is_duplicate_alert(alert):
                            alerts.append(alert)
                            await self._send_alert(alert)
                            
                except Exception as e:
                    logger.error(f"Alert check error: {e}")
        
        # Check market-wide alerts
        if market_impact['risk_level'] == 'high':
            market_alert = {
                'type': 'market_wide',
                'priority': 'high',
                'title': 'High Market Risk Detected',
                'message': f"Overall market impact: {market_impact['overall_score']:.2f}\n" +
                          f"Direction: {market_impact['direction']}\n" +
                          f"Major events: {len(market_impact['major_events'])}"
            }
            alerts.append(market_alert)
            await self._send_alert(market_alert)
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def _create_alert(self, analysis, rule: Dict, market_impact: Dict) -> Dict:
        """Alert oluştur"""
        
        message = rule['message_template'].format(
            title=analysis.title,
            magnitude=analysis.impact.magnitude,
            direction=analysis.impact.expected_direction,
            affected_assets=', '.join(analysis.impact.affected_assets),
            confidence=analysis.impact.confidence,
            sentiment=analysis.sentiment.get('compound', 0),
            source=analysis.source,
            content=analysis.content
        )
        
        return {
            'id': f"{rule['name']}_{datetime.now().timestamp()}",
            'type': rule['name'],
            'priority': rule['priority'],
            'title': analysis.title,
            'message': message,
            'timestamp': datetime.now(),
            'analysis': analysis,
            'market_impact': market_impact
        }
    
    def _is_duplicate_alert(self, alert: Dict) -> bool:
        """Duplicate alert kontrolü"""
        
        # Check last hour's alerts
        cutoff = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > cutoff]
        
        for recent in recent_alerts:
            if recent['type'] == alert['type'] and recent['title'] == alert['title']:
                return True
        
        return False
    
    async def _send_alert(self, alert: Dict):
        """Alert gönder"""
        
        # Send to all configured channels
        for channel in self.notification_channels:
            try:
                if channel == 'telegram':
                    await self._send_telegram_alert(alert)
                elif channel == 'email':
                    await self._send_email_alert(alert)
                elif channel == 'discord':
                    await self._send_discord_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _send_telegram_alert(self, alert: Dict):
        """Telegram alert"""
        # Implementation depends on telegram bot setup
        pass
    
    async def _send_email_alert(self, alert: Dict):
        """Email alert"""
        # Implementation depends on email service
        pass
    
    async def _send_discord_alert(self, alert: Dict):
        """Discord alert"""
        # Implementation depends on discord webhook
        pass

class RedisCache:
    """Redis önbellekleme"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        import redis
        self.client = redis.Redis(host=host, port=port, db=db)
    
    async def get(self, key: str):
        """Cache'den al"""
        try:
            import pickle
            data = self.client.get(key)
            return pickle.loads(data) if data else None
        except:
            return None
    
    async def set(self, key: str, value, ttl: int = 3600):
        """Cache'e kaydet"""
        try:
            import pickle
            self.client.setex(key, ttl, pickle.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

# ============================================
# MAIN EXECUTION
# ============================================

async def main():
    """Ana çalıştırma fonksiyonu"""
    
    # Initialize pipeline
    pipeline = AdvancedDataPipeline()
    
    # Start real-time processing
    await pipeline.process_real_time()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the pipeline
    asyncio.run(main())