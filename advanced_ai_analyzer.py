import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    BertModel,
    RobertaModel,
    DebertaV2Model
)
from sentence_transformers import SentenceTransformer
import spacy
from spacy import displacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import asyncio
import logging
from collections import defaultdict, Counter
import networkx as nx
from textblob import TextBlob
import yfinance as yf
from gensim import corpora, models
from gensim.models import CoherenceModel
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Entity:
    """Tanımlanan varlık (entity) yapısı"""
    text: str
    label: str  # PERSON, ORG, CRYPTO, MONEY, etc.
    start: int
    end: int
    confidence: float
    context: str = ""
    related_entities: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    importance_score: float = 0.0

@dataclass
class Topic:
    """Konu (topic) yapısı"""
    id: int
    name: str
    keywords: List[str]
    weight: float
    documents: List[int]
    coherence_score: float
    trend: str = "stable"  # increasing, decreasing, stable
    sentiment: float = 0.0

@dataclass
class ImpactAnalysis:
    """Etki analizi sonucu"""
    event_type: str
    magnitude: float  # 0-1 scale
    confidence: float
    affected_assets: List[str]
    expected_direction: str  # bullish, bearish, neutral
    time_horizon: str  # immediate, short-term, long-term
    reasoning: str
    historical_accuracy: Optional[float] = None

@dataclass
class NewsAnalysis:
    """Haber analizi sonucu"""
    title: str
    content: str
    entities: List[Entity]
    topics: List[Topic]
    sentiment: Dict[str, float]
    impact: ImpactAnalysis
    credibility_score: float
    virality_potential: float
    timestamp: datetime

# ============================================
# ADVANCED AI ANALYZER
# ============================================

class AdvancedAIAnalyzer:
    """Gelişmiş AI analiz modülü"""
    
    def __init__(self):
        # Load models
        self._initialize_models()
        
        # Initialize components
        self.ner_analyzer = NamedEntityRecognizer()
        self.topic_modeler = TopicModeler()
        self.impact_scorer = ImpactScorer()
        self.sentiment_analyzer = MultiModalSentimentAnalyzer()
        self.credibility_checker = CredibilityChecker()
        
        # Knowledge base
        self.knowledge_graph = KnowledgeGraph()
        self.historical_impacts = []
        
    def _initialize_models(self):
        """Modelleri başlat"""
        logger.info("Initializing AI models...")
        
        # Sentence embeddings
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Zero-shot classification
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Question answering
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        
        # Text generation for reasoning
        self.reasoning_model = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-1.3B",
            max_length=200
        )
        
        logger.info("AI models initialized successfully")
    
    async def analyze_comprehensive(self, text: str, 
                                   metadata: Optional[Dict] = None) -> NewsAnalysis:
        """Kapsamlı analiz yap"""
        
        # 1. Entity Recognition
        entities = await self.ner_analyzer.extract_entities(text)
        
        # 2. Topic Modeling
        topics = await self.topic_modeler.extract_topics([text])
        
        # 3. Sentiment Analysis
        sentiment = await self.sentiment_analyzer.analyze(text, entities)
        
        # 4. Impact Scoring
        impact = await self.impact_scorer.calculate_impact(
            text, entities, sentiment, metadata
        )
        
        # 5. Credibility Check
        credibility = await self.credibility_checker.check(text, metadata)
        
        # 6. Virality Prediction
        virality = self._predict_virality(text, entities, sentiment)
        
        # 7. Update Knowledge Graph
        self.knowledge_graph.update(entities, topics)
        
        return NewsAnalysis(
            title=metadata.get('title', ''),
            content=text,
            entities=entities,
            topics=topics,
            sentiment=sentiment,
            impact=impact,
            credibility_score=credibility,
            virality_potential=virality,
            timestamp=datetime.now()
        )
    
    def _predict_virality(self, text: str, entities: List[Entity], 
                         sentiment: Dict) -> float:
        """Viral olma potansiyelini tahmin et"""
        
        # Factors affecting virality
        factors = {
            'entity_count': len(entities),
            'sentiment_intensity': abs(sentiment.get('compound', 0)),
            'text_length': len(text.split()),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_questions': '?' in text,
            'has_exclamations': '!' in text
        }
        
        # Simple weighted scoring
        weights = {
            'entity_count': 0.2,
            'sentiment_intensity': 0.3,
            'text_length': -0.1,  # Shorter is better
            'has_numbers': 0.2,
            'has_questions': 0.15,
            'has_exclamations': 0.15
        }
        
        score = 0
        for factor, value in factors.items():
            if factor == 'text_length':
                # Normalize text length (optimal around 100 words)
                value = 1 - abs(value - 100) / 100
            elif isinstance(value, bool):
                value = 1 if value else 0
            
            score += value * weights.get(factor, 0)
        
        return max(0, min(1, score))

# ============================================
# NAMED ENTITY RECOGNITION (NER)
# ============================================

class NamedEntityRecognizer:
    """Gelişmiş varlık tanıma sistemi"""
    
    def __init__(self):
        # Load SpaCy model with custom components
        self.nlp = spacy.load("en_core_web_lg")
        
        # Hugging Face NER models
        self.finance_ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
        
        # Custom crypto entity patterns
        self.crypto_patterns = self._load_crypto_patterns()
        
        # Entity linking model
        self.entity_linker = EntityLinker()
    
    def _load_crypto_patterns(self) -> Dict:
        """Kripto varlık pattern'leri"""
        return {
            'CRYPTO': [
                r'\b(BTC|Bitcoin|₿)\b',
                r'\b(ETH|Ethereum|Ether)\b',
                r'\b(BNB|Binance Coin)\b',
                r'\b(ADA|Cardano)\b',
                r'\b(SOL|Solana)\b',
                r'\b(DOT|Polkadot)\b',
                r'\b(DOGE|Dogecoin)\b',
                r'\b(AVAX|Avalanche)\b',
                r'\b(MATIC|Polygon)\b',
                r'\b(LINK|Chainlink)\b'
            ],
            'EXCHANGE': [
                r'\b(Binance|Coinbase|Kraken|FTX|Huobi|KuCoin|Bybit)\b',
                r'\b(Uniswap|SushiSwap|PancakeSwap|Curve)\b'
            ],
            'PERSON': [
                r'\b(Satoshi Nakamoto|Vitalik Buterin|CZ|SBF)\b',
                r'\b(Elon Musk|Michael Saylor|Cathie Wood)\b'
            ],
            'ORGANIZATION': [
                r'\b(SEC|CFTC|Fed|Federal Reserve|ECB)\b',
                r'\b(Tesla|MicroStrategy|Square|PayPal)\b',
                r'\b(JP Morgan|Goldman Sachs|Morgan Stanley)\b'
            ],
            'EVENT': [
                r'\b(halving|hard fork|soft fork|airdrop)\b',
                r'\b(ICO|IDO|IEO|token sale)\b',
                r'\b(hack|exploit|rug pull|exit scam)\b'
            ],
            'TECHNICAL': [
                r'\b(blockchain|DeFi|NFT|Web3|metaverse)\b',
                r'\b(smart contract|DAO|DApp|layer 2)\b',
                r'\b(proof of work|proof of stake|consensus)\b'
            ]
        }
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Varlıkları çıkar"""
        entities = []
        
        # 1. SpaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.9,
                context=text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
            )
            entities.append(entity)
        
        # 2. Hugging Face Finance NER
        hf_entities = self.finance_ner(text)
        for ent in hf_entities:
            entity = Entity(
                text=ent['word'],
                label=ent['entity_group'],
                start=ent['start'],
                end=ent['end'],
                confidence=ent['score'],
                context=text[max(0, ent['start']-50):min(len(text), ent['end']+50)]
            )
            entities.append(entity)
        
        # 3. Custom Crypto Pattern Matching
        for label, patterns in self.crypto_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    )
                    entities.append(entity)
        
        # 4. Entity Disambiguation and Linking
        entities = await self.entity_linker.link_entities(entities)
        
        # 5. Calculate entity importance
        entities = self._calculate_entity_importance(entities, text)
        
        # Remove duplicates
        unique_entities = self._deduplicate_entities(entities)
        
        return unique_entities
    
    def _calculate_entity_importance(self, entities: List[Entity], 
                                    text: str) -> List[Entity]:
        """Varlık önem skorunu hesapla"""
        
        # Factors for importance
        text_lower = text.lower()
        
        for entity in entities:
            # Frequency
            frequency = text_lower.count(entity.text.lower())
            
            # Position (earlier = more important)
            position_score = 1 - (entity.start / len(text))
            
            # Title presence
            title_presence = 1.0 if entity.start < 100 else 0.5
            
            # Entity type weight
            type_weights = {
                'CRYPTO': 1.0,
                'MONEY': 0.9,
                'PERSON': 0.8,
                'ORGANIZATION': 0.8,
                'EVENT': 0.7,
                'EXCHANGE': 0.9
            }
            type_weight = type_weights.get(entity.label, 0.5)
            
            # Calculate importance
            entity.importance_score = (
                frequency * 0.3 +
                position_score * 0.2 +
                title_presence * 0.2 +
                type_weight * 0.3
            )
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Duplicate varlıkları temizle"""
        seen = set()
        unique = []
        
        for entity in sorted(entities, key=lambda x: x.confidence, reverse=True):
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique

class EntityLinker:
    """Entity linking ve disambiguation"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict:
        """Knowledge base yükle"""
        return {
            'BTC': {'full_name': 'Bitcoin', 'type': 'cryptocurrency', 'wiki': 'Bitcoin'},
            'ETH': {'full_name': 'Ethereum', 'type': 'cryptocurrency', 'wiki': 'Ethereum'},
            'SEC': {'full_name': 'Securities and Exchange Commission', 'type': 'regulator'},
            'Elon': {'full_name': 'Elon Musk', 'type': 'person', 'role': 'CEO of Tesla'},
            # Add more entities
        }
    
    async def link_entities(self, entities: List[Entity]) -> List[Entity]:
        """Varlıkları knowledge base'e bağla"""
        
        for entity in entities:
            # Simple string matching for now
            for key, info in self.knowledge_base.items():
                if key.lower() in entity.text.lower():
                    entity.related_entities.append(info['full_name'])
                    # Add more metadata if available
        
        return entities

# ============================================
# TOPIC MODELING
# ============================================

class TopicModeler:
    """Gelişmiş konu modelleme sistemi"""
    
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        
        # Different topic modeling approaches
        self.lda_model = None
        self.nmf_model = None
        self.bert_topic = None
        
        # Preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        self.topic_history = []
    
    async def extract_topics(self, documents: List[str],
                           method: str = 'hybrid') -> List[Topic]:
        """Konuları çıkar"""
        
        if method == 'lda':
            topics = await self._lda_topic_modeling(documents)
        elif method == 'nmf':
            topics = await self._nmf_topic_modeling(documents)
        elif method == 'bertopic':
            topics = await self._bert_topic_modeling(documents)
        else:  # hybrid
            topics = await self._hybrid_topic_modeling(documents)
        
        # Analyze topic trends
        topics = self._analyze_topic_trends(topics)
        
        # Calculate topic coherence
        topics = self._calculate_coherence(topics, documents)
        
        return topics
    
    async def _lda_topic_modeling(self, documents: List[str]) -> List[Topic]:
        """Latent Dirichlet Allocation"""
        
        # Vectorize documents
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='batch'
        )
        
        lda_output = self.lda_model.fit_transform(doc_term_matrix)
        
        # Extract topics
        topics = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            
            topic_obj = Topic(
                id=topic_idx,
                name=f"Topic_{topic_idx}",
                keywords=top_features,
                weight=topic.mean(),
                documents=[],
                coherence_score=0.0
            )
            topics.append(topic_obj)
        
        return topics
    
    async def _nmf_topic_modeling(self, documents: List[str]) -> List[Topic]:
        """Non-negative Matrix Factorization"""
        
        # Vectorize
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # NMF model
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            init='nndsvd'
        )
        
        nmf_output = self.nmf_model.fit_transform(doc_term_matrix)
        
        # Extract topics
        topics = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            
            topic_obj = Topic(
                id=topic_idx,
                name=self._generate_topic_name(top_features),
                keywords=top_features,
                weight=topic.mean(),
                documents=[],
                coherence_score=0.0
            )
            topics.append(topic_obj)
        
        return topics
    
    async def _bert_topic_modeling(self, documents: List[str]) -> List[Topic]:
        """BERT-based topic modeling"""
        try:
            from bertopic import BERTopic
            
            # BERTopic model
            self.bert_topic = BERTopic(
                n_gram_range=(1, 3),
                min_topic_size=2,
                diversity=0.5
            )
            
            topics_bert, probs = self.bert_topic.fit_transform(documents)
            
            # Extract topic information
            topics = []
            topic_info = self.bert_topic.get_topic_info()
            
            for index, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    topic_obj = Topic(
                        id=row['Topic'],
                        name=row['Name'],
                        keywords=self.bert_topic.get_topic(row['Topic'])[:10],
                        weight=row['Count'] / len(documents),
                        documents=[],
                        coherence_score=0.0
                    )
                    topics.append(topic_obj)
            
            return topics
            
        except ImportError:
            logger.warning("BERTopic not installed, falling back to LDA")
            return await self._lda_topic_modeling(documents)
    
    async def _hybrid_topic_modeling(self, documents: List[str]) -> List[Topic]:
        """Hybrid approach combining multiple methods"""
        
        # Get topics from different methods
        lda_topics = await self._lda_topic_modeling(documents)
        nmf_topics = await self._nmf_topic_modeling(documents)
        
        # Merge and reconcile topics
        merged_topics = self._merge_topics(lda_topics, nmf_topics)
        
        return merged_topics
    
    def _merge_topics(self, topics1: List[Topic], 
                     topics2: List[Topic]) -> List[Topic]:
        """İki topic setini birleştir"""
        
        merged = []
        used_indices = set()
        
        for t1 in topics1:
            best_match = None
            best_similarity = 0
            
            for i, t2 in enumerate(topics2):
                if i not in used_indices:
                    # Calculate keyword overlap
                    overlap = len(set(t1.keywords) & set(t2.keywords))
                    similarity = overlap / max(len(t1.keywords), len(t2.keywords))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (i, t2)
            
            if best_match and best_similarity > 0.3:
                # Merge topics
                idx, t2 = best_match
                used_indices.add(idx)
                
                merged_topic = Topic(
                    id=len(merged),
                    name=t1.name,
                    keywords=list(set(t1.keywords + t2.keywords))[:15],
                    weight=(t1.weight + t2.weight) / 2,
                    documents=list(set(t1.documents + t2.documents)),
                    coherence_score=0.0
                )
                merged.append(merged_topic)
            else:
                merged.append(t1)
        
        return merged
    
    def _generate_topic_name(self, keywords: List[str]) -> str:
        """Topic ismi oluştur"""
        
        # Simple heuristic: use top 3 keywords
        if len(keywords) >= 3:
            return f"{keywords[0]}_{keywords[1]}_{keywords[2]}"
        else:
            return "_".join(keywords)
    
    def _analyze_topic_trends(self, topics: List[Topic]) -> List[Topic]:
        """Topic trend'lerini analiz et"""
        
        for topic in topics:
            # Compare with historical topics
            if self.topic_history:
                historical_weights = [
                    h.get(topic.id, 0) 
                    for h in self.topic_history[-5:]
                ]
                
                if historical_weights:
                    recent_avg = np.mean(historical_weights[-2:])
                    older_avg = np.mean(historical_weights[:-2])
                    
                    if recent_avg > older_avg * 1.2:
                        topic.trend = "increasing"
                    elif recent_avg < older_avg * 0.8:
                        topic.trend = "decreasing"
                    else:
                        topic.trend = "stable"
        
        # Save current topics to history
        self.topic_history.append({t.id: t.weight for t in topics})
        
        return topics
    
    def _calculate_coherence(self, topics: List[Topic], 
                            documents: List[str]) -> List[Topic]:
        """Topic coherence hesapla"""
        
        # Simple coherence based on word co-occurrence
        for topic in topics:
            coherence = 0
            keyword_pairs = 0
            
            for i, word1 in enumerate(topic.keywords):
                for word2 in topic.keywords[i+1:]:
                    # Count co-occurrences
                    co_occur = sum(
                        1 for doc in documents 
                        if word1 in doc and word2 in doc
                    )
                    
                    # Normalize by total documents
                    coherence += co_occur / len(documents)
                    keyword_pairs += 1
            
            if keyword_pairs > 0:
                topic.coherence_score = coherence / keyword_pairs
        
        return topics

# ============================================
# IMPACT SCORING
# ============================================

class ImpactScorer:
    """Etki skorlama sistemi"""
    
    def __init__(self):
        self.historical_impacts = self._load_historical_impacts()
        self.market_correlations = self._load_market_correlations()
        
        # Event classifier
        self.event_classifier = EventClassifier()
        
        # Market predictor
        self.market_predictor = MarketImpactPredictor()
    
    def _load_historical_impacts(self) -> pd.DataFrame:
        """Tarihi etki verilerini yükle"""
        # In production, load from database
        return pd.DataFrame()
    
    def _load_market_correlations(self) -> Dict:
        """Piyasa korelasyonlarını yükle"""
        return {
            'regulation': {'BTC': -0.3, 'ETH': -0.25},
            'adoption': {'BTC': 0.4, 'ETH': 0.35},
            'hack': {'affected_token': -0.5, 'market': -0.1},
            'partnership': {'affected_token': 0.3, 'market': 0.05}
        }
    
    async def calculate_impact(self, text: str, entities: List[Entity],
                              sentiment: Dict, metadata: Optional[Dict]) -> ImpactAnalysis:
        """Etki skorunu hesapla"""
        
        # 1. Classify event type
        event_type = await self.event_classifier.classify(text)
        
        # 2. Extract affected assets
        affected_assets = self._extract_affected_assets(entities)
        
        # 3. Calculate magnitude
        magnitude = await self._calculate_magnitude(
            text, event_type, sentiment, entities
        )
        
        # 4. Predict market impact
        market_impact = await self.market_predictor.predict(
            event_type, magnitude, affected_assets
        )
        
        # 5. Generate reasoning
        reasoning = await self._generate_reasoning(
            text, event_type, magnitude, market_impact
        )
        
        # 6. Calculate confidence
        confidence = self._calculate_confidence(
            entities, sentiment, metadata
        )
        
        # 7. Check historical accuracy
        historical_accuracy = self._check_historical_accuracy(
            event_type, magnitude
        )
        
        return ImpactAnalysis(
            event_type=event_type,
            magnitude=magnitude,
            confidence=confidence,
            affected_assets=affected_assets,
            expected_direction=market_impact['direction'],
            time_horizon=market_impact['time_horizon'],
            reasoning=reasoning,
            historical_accuracy=historical_accuracy
        )
    
    def _extract_affected_assets(self, entities: List[Entity]) -> List[str]:
        """Etkilenen varlıkları çıkar"""
        affected = []
        
        for entity in entities:
            if entity.label in ['CRYPTO', 'EXCHANGE', 'ORGANIZATION']:
                affected.append(entity.text)
        
        return list(set(affected))
    
    async def _calculate_magnitude(self, text: str, event_type: str,
                                  sentiment: Dict, entities: List[Entity]) -> float:
        """Etki büyüklüğünü hesapla"""
        
        # Base magnitude from event type
        event_magnitudes = {
            'regulation': 0.7,
            'hack': 0.8,
            'adoption': 0.6,
            'partnership': 0.5,
            'technical_update': 0.4,
            'market_movement': 0.3
        }
        base_magnitude = event_magnitudes.get(event_type, 0.3)
        
        # Adjust based on sentiment intensity
        sentiment_factor = abs(sentiment.get('compound', 0))
        
        # Entity importance factor
        entity_factor = max([e.importance_score for e in entities]) if entities else 0.5
        
        # Text length factor (longer = more detailed = potentially more important)
        length_factor = min(len(text.split()) / 500, 1.0)
        
        # Calculate weighted magnitude
        magnitude = (
            base_magnitude * 0.4 +
            sentiment_factor * 0.3 +
            entity_factor * 0.2 +
            length_factor * 0.1
        )
        
        return max(0, min(1, magnitude))
    
    async def _generate_reasoning(self, text: str, event_type: str,
                                 magnitude: float, market_impact: Dict) -> str:
        """Akıl yürütme oluştur"""
        
        # Use language model to generate reasoning
        prompt = f"""
        Event Type: {event_type}
        Magnitude: {magnitude:.2f}
        Expected Direction: {market_impact['direction']}
        
        Based on the following text, explain the potential market impact:
        {text[:500]}
        
        Reasoning:
        """
        
        try:
            # Use reasoning model if available
            # response = self.reasoning_model(prompt)
            # return response[0]['generated_text']
            
            # Fallback to template-based reasoning
            reasoning = f"This {event_type} event with magnitude {magnitude:.2f} "
            reasoning += f"is expected to have a {market_impact['direction']} impact "
            reasoning += f"on the market in the {market_impact['time_horizon']}."
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return "Impact analysis based on historical patterns and current market conditions."
    
    def _calculate_confidence(self, entities: List[Entity],
                             sentiment: Dict, metadata: Optional[Dict]) -> float:
        """Güven skorunu hesapla"""
        
        confidence = 0.5  # Base confidence
        
        # Entity confidence
        if entities:
            avg_entity_confidence = np.mean([e.confidence for e in entities])
            confidence += avg_entity_confidence * 0.2
        
        # Source credibility
        if metadata and 'source' in metadata:
            source_credibility = {
                'reuters': 0.9,
                'bloomberg': 0.9,
                'coindesk': 0.8,
                'cointelegraph': 0.7,
                'twitter': 0.5,
                'reddit': 0.4
            }
            confidence *= source_credibility.get(metadata['source'].lower(), 0.5)
        
        # Historical accuracy bonus
        if hasattr(self, 'historical_accuracy') and self.historical_accuracy > 0.7:
            confidence += 0.1
        
        return max(0, min(1, confidence))
    
    def _check_historical_accuracy(self, event_type: str, 
                                  magnitude: float) -> Optional[float]:
        """Tarihi doğruluğu kontrol et"""
        
        if self.historical_impacts.empty:
            return None
        
        # Filter similar events
        similar_events = self.historical_impacts[
            (self.historical_impacts['event_type'] == event_type) &
            (abs(self.historical_impacts['magnitude'] - magnitude) < 0.2)
        ]
        
        if len(similar_events) > 0:
            # Calculate accuracy
            correct_predictions = similar_events['correct_prediction'].sum()
            total_predictions = len(similar_events)
            
            return correct_predictions / total_predictions
        
        return None

class EventClassifier:
    """Olay sınıflandırıcı"""
    
    def __init__(self):
        self.event_categories = [
            'regulation',
            'hack',
            'adoption',
            'partnership',
            'technical_update',
            'market_movement',
            'announcement',
            'legal',
            'economic_data'
        ]
        
        # Keywords for each category
        self.event_keywords = {
            'regulation': ['sec', 'regulate', 'ban', 'approve', 'policy', 'law'],
            'hack': ['hack', 'breach', 'exploit', 'attack', 'stolen', 'vulnerability'],
            'adoption': ['adopt', 'accept', 'integrate', 'launch', 'implement'],
            'partnership': ['partner', 'collaborate', 'alliance', 'deal', 'agreement'],
            'technical_update': ['upgrade', 'fork', 'update', 'release', 'mainnet'],
            'market_movement': ['surge', 'crash', 'rally', 'dump', 'pump', 'volatility'],
            'announcement': ['announce', 'reveal', 'disclose', 'confirm'],
            'legal': ['lawsuit', 'sue', 'court', 'legal', 'settlement'],
            'economic_data': ['inflation', 'gdp', 'employment', 'fed', 'interest rate']
        }
    
    async def classify(self, text: str) -> str:
        """Olayı sınıflandır"""
        
        text_lower = text.lower()
        scores = {}
        
        # Keyword-based scoring
        for category, keywords in self.event_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        # Get category with highest score
        if scores:
            return max(scores, key=scores.get)
        
        return 'announcement'  # Default

class MarketImpactPredictor:
    """Piyasa etkisi tahmin edici"""
    
    def __init__(self):
        self.impact_patterns = self._load_impact_patterns()
    
    def _load_impact_patterns(self) -> Dict:
        """Etki pattern'lerini yükle"""
        return {
            'regulation': {
                'positive': ['approve', 'clarity', 'framework'],
                'negative': ['ban', 'restrict', 'investigate']
            },
            'hack': {
                'negative': ['stolen', 'breach', 'exploit']
            },
            'adoption': {
                'positive': ['accept', 'integrate', 'launch']
            }
        }
    
    async def predict(self, event_type: str, magnitude: float,
                     affected_assets: List[str]) -> Dict:
        """Piyasa etkisini tahmin et"""
        
        # Determine direction based on event type
        direction_map = {
            'regulation': 'bearish',  # Generally negative initially
            'hack': 'bearish',
            'adoption': 'bullish',
            'partnership': 'bullish',
            'technical_update': 'neutral',
            'market_movement': 'neutral'  # Depends on context
        }
        
        direction = direction_map.get(event_type, 'neutral')
        
        # Determine time horizon based on magnitude
        if magnitude > 0.7:
            time_horizon = 'immediate'
        elif magnitude > 0.4:
            time_horizon = 'short-term'
        else:
            time_horizon = 'long-term'
        
        return {
            'direction': direction,
            'time_horizon': time_horizon,
            'confidence': magnitude
        }

# ============================================
# MULTI-MODAL SENTIMENT ANALYZER
# ============================================

class MultiModalSentimentAnalyzer:
    """Çok modlu duygu analizi"""
    
    def __init__(self):
        # Different sentiment models
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        
        self.crypto_bert = pipeline(
            "sentiment-analysis",
            model="ElKulako/cryptobert"
        )
        
        self.general_sentiment = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Aspect-based sentiment
        self.aspect_sentiment = AspectSentimentAnalyzer()
    
    async def analyze(self, text: str, entities: List[Entity]) -> Dict:
        """Çok boyutlu sentiment analizi"""
        
        # 1. General sentiment
        general = self._analyze_general(text)
        
        # 2. Financial sentiment
        financial = self._analyze_financial(text)
        
        # 3. Crypto-specific sentiment
        crypto = self._analyze_crypto(text)
        
        # 4. Entity-specific sentiment
        entity_sentiments = await self._analyze_entity_sentiments(text, entities)
        
        # 5. Aspect-based sentiment
        aspects = await self.aspect_sentiment.analyze(text)
        
        # Combine all sentiments
        combined = {
            'general': general,
            'financial': financial,
            'crypto': crypto,
            'entities': entity_sentiments,
            'aspects': aspects,
            'compound': self._calculate_compound_sentiment({
                'general': general,
                'financial': financial,
                'crypto': crypto
            })
        }
        
        return combined
    
    def _analyze_general(self, text: str) -> float:
        """Genel sentiment analizi"""
        try:
            result = self.general_sentiment(text[:512])
            
            if result[0]['label'] == 'POSITIVE':
                return result[0]['score']
            elif result[0]['label'] == 'NEGATIVE':
                return -result[0]['score']
            else:
                return 0
        except:
            # Fallback to TextBlob
            return TextBlob(text).sentiment.polarity
    
    def _analyze_financial(self, text: str) -> float:
        """Finansal sentiment analizi"""
        try:
            result = self.finbert(text[:512])
            
            if result[0]['label'] == 'positive':
                return result[0]['score']
            elif result[0]['label'] == 'negative':
                return -result[0]['score']
            else:
                return 0
        except:
            return 0
    
    def _analyze_crypto(self, text: str) -> float:
        """Kripto-spesifik sentiment analizi"""
        try:
            result = self.crypto_bert(text[:512])
            
            # Process crypto-specific sentiment
            return result[0]['score'] if result[0]['label'] == 'Bullish' else -result[0]['score']
        except:
            # Fallback to keyword-based
            bullish_keywords = ['moon', 'bullish', 'pump', 'hodl', 'buy']
            bearish_keywords = ['dump', 'bearish', 'crash', 'sell', 'short']
            
            text_lower = text.lower()
            bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
            bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
            
            return (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
    
    async def _analyze_entity_sentiments(self, text: str, 
                                        entities: List[Entity]) -> Dict:
        """Her entity için sentiment analizi"""
        
        entity_sentiments = {}
        
        for entity in entities:
            # Get context around entity
            context = entity.context or text
            
            # Analyze sentiment for this context
            sentiment = self._analyze_general(context)
            
            entity.sentiment = sentiment
            entity_sentiments[entity.text] = sentiment
        
        return entity_sentiments
    
    def _calculate_compound_sentiment(self, sentiments: Dict) -> float:
        """Bileşik sentiment hesapla"""
        
        # Weighted average
        weights = {
            'general': 0.3,
            'financial': 0.4,
            'crypto': 0.3
        }
        
        compound = sum(
            sentiments.get(key, 0) * weight 
            for key, weight in weights.items()
        )
        
        return max(-1, min(1, compound))

class AspectSentimentAnalyzer:
    """Aspect-based sentiment analysis"""
    
    def __init__(self):
        self.aspects = [
            'price',
            'technology',
            'adoption',
            'regulation',
            'team',
            'community'
        ]
    
    async def analyze(self, text: str) -> Dict:
        """Her aspect için sentiment analizi"""
        
        aspect_sentiments = {}
        
        for aspect in self.aspects:
            # Find sentences mentioning this aspect
            sentences = self._find_aspect_sentences(text, aspect)
            
            if sentences:
                # Analyze sentiment for these sentences
                sentiments = [TextBlob(s).sentiment.polarity for s in sentences]
                aspect_sentiments[aspect] = np.mean(sentiments)
            else:
                aspect_sentiments[aspect] = 0
        
        return aspect_sentiments
    
    def _find_aspect_sentences(self, text: str, aspect: str) -> List[str]:
        """Aspect'i içeren cümleleri bul"""
        
        aspect_keywords = {
            'price': ['price', 'value', 'cost', 'worth', 'expensive', 'cheap'],
            'technology': ['tech', 'blockchain', 'smart contract', 'protocol', 'code'],
            'adoption': ['adopt', 'use', 'accept', 'integrate', 'implement'],
            'regulation': ['regulate', 'law', 'sec', 'legal', 'compliance'],
            'team': ['team', 'founder', 'developer', 'ceo', 'leadership'],
            'community': ['community', 'users', 'holders', 'supporters', 'fans']
        }
        
        keywords = aspect_keywords.get(aspect, [aspect])
        sentences = text.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence)
        
        return relevant_sentences

# ============================================
# CREDIBILITY CHECKER
# ============================================

class CredibilityChecker:
    """Kaynak güvenilirlik kontrolü"""
    
    def __init__(self):
        self.fact_checker = FactChecker()
        self.source_ranker = SourceRanker()
    
    async def check(self, text: str, metadata: Optional[Dict]) -> float:
        """Güvenilirlik kontrolü yap"""
        
        credibility = 0.5  # Base credibility
        
        # 1. Source credibility
        if metadata and 'source' in metadata:
            source_score = self.source_ranker.rank(metadata['source'])
            credibility = credibility * 0.3 + source_score * 0.7
        
        # 2. Fact checking
        fact_score = await self.fact_checker.check_facts(text)
        credibility = credibility * 0.7 + fact_score * 0.3
        
        # 3. Writing quality indicators
        quality_score = self._assess_writing_quality(text)
        credibility = credibility * 0.8 + quality_score * 0.2
        
        return max(0, min(1, credibility))
    
    def _assess_writing_quality(self, text: str) -> float:
        """Yazım kalitesini değerlendir"""
        
        quality = 1.0
        
        # Check for excessive capitals
        if text.isupper():
            quality -= 0.3
        
        # Check for excessive punctuation
        if text.count('!') > 3 or text.count('?') > 3:
            quality -= 0.2
        
        # Check for grammar (simplified)
        blob = TextBlob(text)
        try:
            # Spelling mistakes
            if blob.correct() != blob:
                quality -= 0.1
        except:
            pass
        
        # Check for clickbait patterns
        clickbait_patterns = [
            'you won\'t believe',
            'shocking',
            'breaking',
            'exclusive'
        ]
        
        text_lower = text.lower()
        for pattern in clickbait_patterns:
            if pattern in text_lower:
                quality -= 0.1
        
        return max(0, quality)

class FactChecker:
    """Fact checking sistem"""
    
    async def check_facts(self, text: str) -> float:
        """Fact'leri kontrol et"""
        
        # Extract claims
        claims = self._extract_claims(text)
        
        if not claims:
            return 0.7  # Neutral if no specific claims
        
        # Check each claim
        verified = 0
        for claim in claims:
            if await self._verify_claim(claim):
                verified += 1
        
        return verified / len(claims) if claims else 0.7
    
    def _extract_claims(self, text: str) -> List[str]:
        """İddiaları çıkar"""
        
        claims = []
        
        # Look for numerical claims
        number_pattern = r'\d+(?:\.\d+)?%?'
        matches = re.findall(f'.*{number_pattern}.*', text)
        claims.extend(matches[:5])  # Limit to 5 claims
        
        return claims
    
    async def _verify_claim(self, claim: str) -> bool:
        """İddiayı doğrula"""
        
        # Simplified verification
        # In production, would check against reliable sources
        return True  # Placeholder

class SourceRanker:
    """Kaynak güvenilirlik sıralaması"""
    
    def __init__(self):
        self.source_rankings = {
            # Tier 1 - Most reliable
            'reuters': 0.95,
            'bloomberg': 0.95,
            'wsj': 0.93,
            'ft': 0.93,
            
            # Tier 2 - Reliable crypto sources
            'coindesk': 0.85,
            'theblock': 0.83,
            'decrypt': 0.80,
            
            # Tier 3 - General crypto news
            'cointelegraph': 0.75,
            'bitcoinmagazine': 0.73,
            'cryptonews': 0.70,
            
            # Tier 4 - Social media
            'twitter': 0.50,
            'reddit': 0.45,
            'telegram': 0.40,
            
            # Tier 5 - Unknown
            'unknown': 0.30
        }
    
    def rank(self, source: str) -> float:
        """Kaynak güvenilirlik skorunu al"""
        
        source_lower = source.lower()
        
        for key, score in self.source_rankings.items():
            if key in source_lower:
                return score
        
        return self.source_rankings['unknown']

# ============================================
# KNOWLEDGE GRAPH
# ============================================

class KnowledgeGraph:
    """Bilgi grafiği"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
    
    def update(self, entities: List[Entity], topics: List[Topic]):
        """Bilgi grafiğini güncelle"""
        
        # Add entities as nodes
        for entity in entities:
            if not self.graph.has_node(entity.text):
                self.graph.add_node(
                    entity.text,
                    type=entity.label,
                    importance=entity.importance_score,
                    sentiment=entity.sentiment
                )
            else:
                # Update existing node
                node = self.graph.nodes[entity.text]
                node['importance'] = (node['importance'] + entity.importance_score) / 2
                node['sentiment'] = (node['sentiment'] + entity.sentiment) / 2
        
        # Create edges between co-occurring entities
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                if not self.graph.has_edge(e1.text, e2.text):
                    self.graph.add_edge(e1.text, e2.text, weight=1)
                else:
                    # Increase edge weight
                    self.graph[e1.text][e2.text]['weight'] += 1
        
        # Add topic relationships
        for topic in topics:
            topic_node = f"Topic_{topic.id}"
            if not self.graph.has_node(topic_node):
                self.graph.add_node(
                    topic_node,
                    type='TOPIC',
                    keywords=topic.keywords
                )
            
            # Connect entities to topics
            for entity in entities:
                if any(keyword in entity.context for keyword in topic.keywords):
                    self.graph.add_edge(entity.text, topic_node)
    
    def get_related_entities(self, entity: str, depth: int = 2) -> List[str]:
        """İlgili varlıkları al"""
        
        if entity not in self.graph:
            return []
        
        related = []
        visited = set()
        queue = [(entity, 0)]
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if current in visited or current_depth > depth:
                continue
            
            visited.add(current)
            
            if current != entity:
                related.append(current)
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
        
        return related
    
    def get_important_entities(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """En önemli varlıkları al"""
        
        # Calculate PageRank
        pagerank = nx.pagerank(self.graph)
        
        # Sort by importance
        sorted_entities = sorted(
            pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_entities[:top_k]
    
    def find_communities(self) -> List[List[str]]:
        """Toplulukları bul"""
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Find communities using Louvain method
        import community as community_louvain
        
        try:
            partition = community_louvain.best_partition(undirected)
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            return list(communities.values())
        except:
            # Fallback to connected components
            components = nx.connected_components(undirected)
            return list(components)

# ============================================
# AI ORCHESTRATOR
# ============================================

class AIOrchestrator:
    """AI modüllerini orkestre eden ana sınıf"""
    
    def __init__(self):
        self.analyzer = AdvancedAIAnalyzer()
        self.report_generator = ReportGenerator()
        self.alert_system = AIAlertSystem()
        
    async def process_news_stream(self, news_items: List[Dict]) -> Dict:
        """Haber akışını işle"""
        
        results = {
            'analyses': [],
            'alerts': [],
            'market_impact': {},
            'report': None
        }
        
        # Analyze each news item
        for item in news_items:
            analysis = await self.analyzer.analyze_comprehensive(
                item['content'],
                item.get('metadata')
            )
            results['analyses'].append(analysis)
            
            # Check for alerts
            if analysis.impact.magnitude > 0.7:
                alert = self.alert_system.create_alert(analysis)
                results['alerts'].append(alert)
        
        # Aggregate market impact
        results['market_impact'] = self._aggregate_market_impact(
            results['analyses']
        )
        
        # Generate report
        results['report'] = await self.report_generator.generate(
            results['analyses']
        )
        
        return results
    
    def _aggregate_market_impact(self, analyses: List[NewsAnalysis]) -> Dict:
        """Toplam piyasa etkisini hesapla"""
        
        impact = {
            'overall_sentiment': 0,
            'expected_movement': 0,
            'confidence': 0,
            'affected_sectors': defaultdict(float)
        }
        
        for analysis in analyses:
            # Weight by importance and credibility
            weight = analysis.impact.magnitude * analysis.credibility_score
            
            impact['overall_sentiment'] += analysis.sentiment.get('compound', 0) * weight
            
            # Expected movement
            if analysis.impact.expected_direction == 'bullish':
                impact['expected_movement'] += weight
            elif analysis.impact.expected_direction == 'bearish':
                impact['expected_movement'] -= weight
            
            impact['confidence'] += analysis.impact.confidence * weight
            
            # Affected sectors
            for asset in analysis.impact.affected_assets:
                impact['affected_sectors'][asset] += weight
        
        # Normalize
        total_weight = sum(a.impact.magnitude * a.credibility_score for a in analyses)
        if total_weight > 0:
            impact['overall_sentiment'] /= total_weight
            impact['expected_movement'] /= total_weight
            impact['confidence'] /= total_weight
        
        return impact

class ReportGenerator:
    """AI rapor üretici"""
    
    async def generate(self, analyses: List[NewsAnalysis]) -> str:
        """Detaylı rapor oluştur"""
        
        report = "🤖 AI ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Executive Summary
        report += "📋 EXECUTIVE SUMMARY\n"
        report += "-" * 30 + "\n"
        
        # Key findings
        top_impacts = sorted(
            analyses,
            key=lambda x: x.impact.magnitude,
            reverse=True
        )[:3]
        
        for analysis in top_impacts:
            report += f"• {analysis.title[:50]}...\n"
            report += f"  Impact: {analysis.impact.magnitude:.2f} "
            report += f"({analysis.impact.expected_direction})\n"
            report += f"  Confidence: {analysis.impact.confidence:.2%}\n\n"
        
        # Entity Analysis
        report += "\n👥 KEY ENTITIES\n"
        report += "-" * 30 + "\n"
        
        all_entities = []
        for analysis in analyses:
            all_entities.extend(analysis.entities)
        
        # Count entity frequencies
        entity_counts = Counter([e.text for e in all_entities])
        
        for entity, count in entity_counts.most_common(5):
            report += f"• {entity}: {count} mentions\n"
        
        # Topic Trends
        report += "\n📊 TRENDING TOPICS\n"
        report += "-" * 30 + "\n"
        
        all_topics = []
        for analysis in analyses:
            all_topics.extend(analysis.topics)
        
        # Group by topic name
        topic_groups = defaultdict(list)
        for topic in all_topics:
            topic_groups[topic.name].append(topic)
        
        for topic_name, topics in list(topic_groups.items())[:5]:
            avg_weight = np.mean([t.weight for t in topics])
            report += f"• {topic_name}: weight={avg_weight:.2f}\n"
        
        return report

class AIAlertSystem:
    """AI tabanlı alert sistemi"""
    
    def create_alert(self, analysis: NewsAnalysis) -> Dict:
        """Alert oluştur"""
        
        alert_level = self._determine_alert_level(analysis)
        
        return {
            'level': alert_level,
            'title': analysis.title,
            'impact': analysis.impact.magnitude,
            'direction': analysis.impact.expected_direction,
            'confidence': analysis.impact.confidence,
            'affected_assets': analysis.impact.affected_assets,
            'reasoning': analysis.impact.reasoning,
            'timestamp': analysis.timestamp
        }
    
    def _determine_alert_level(self, analysis: NewsAnalysis) -> str:
        """Alert seviyesini belirle"""
        
        if analysis.impact.magnitude > 0.8:
            return 'CRITICAL'
        elif analysis.impact.magnitude > 0.6:
            return 'HIGH'
        elif analysis.impact.magnitude > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW' '