"""
DETECTAR.AI - AI Content Detection API
Enterprise-grade detection engine for Spanish & English content
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import math
import re
import statistics
from collections import Counter
import unicodedata

app = FastAPI(
    title="DETECTAR.AI",
    description="AI Content Detection API - EU AI Act Compliant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50000)
    language: Optional[str] = Field(default="auto", description="auto, en, es")

class DetectionSignal(BaseModel):
    name: str
    score: float
    weight: float
    description: str

class DetectionResponse(BaseModel):
    id: str
    timestamp: str
    ai_probability: float
    confidence: str
    verdict: str
    signals: List[DetectionSignal]
    language_detected: str
    word_count: int
    character_count: int
    analysis_time_ms: int

class ScanHistoryItem(BaseModel):
    id: str
    timestamp: str
    text_preview: str
    ai_probability: float
    verdict: str
    word_count: int

# In-memory storage for MVP (replace with DB in production)
scan_history: List[Dict[str, Any]] = []

# ============================================================================
# DETECTION ENGINE
# ============================================================================

class AIDetectionEngine:
    """
    Multi-signal AI detection using statistical analysis.
    Combines multiple heuristics to estimate AI-generated probability.
    """
    
    # Common AI writing patterns
    AI_PHRASES = {
        'en': [
            'it is important to note', 'it\'s worth noting', 'in conclusion',
            'furthermore', 'moreover', 'additionally', 'in summary',
            'it is essential', 'it is crucial', 'play a crucial role',
            'in today\'s world', 'in the realm of', 'delve into',
            'landscape', 'leverage', 'utilize', 'facilitate',
            'comprehensive', 'robust', 'seamless', 'cutting-edge',
            'it is worth mentioning', 'as we can see', 'as mentioned earlier',
            'in this article', 'let\'s explore', 'dive deep',
            'game-changer', 'paradigm shift', 'synergy',
        ],
        'es': [
            'es importante destacar', 'cabe mencionar', 'en conclusión',
            'además', 'asimismo', 'por otro lado', 'en resumen',
            'es esencial', 'es crucial', 'juega un papel crucial',
            'en el mundo actual', 'en el ámbito de', 'profundizar en',
            'panorama', 'aprovechar', 'utilizar', 'facilitar',
            'integral', 'robusto', 'sin fisuras', 'de vanguardia',
            'vale la pena mencionar', 'como podemos ver', 'como se mencionó',
            'en este artículo', 'exploremos', 'ahondar en',
        ]
    }
    
    # Sentence starters typical of AI
    AI_STARTERS = {
        'en': [
            'This', 'It', 'The', 'However', 'Furthermore', 'Moreover',
            'Additionally', 'In', 'As', 'While', 'When', 'Although',
        ],
        'es': [
            'Este', 'Esto', 'El', 'La', 'Sin embargo', 'Además',
            'Asimismo', 'En', 'Como', 'Mientras', 'Cuando', 'Aunque',
        ]
    }
    
    def __init__(self):
        self.weights = {
            'vocabulary_diversity': 0.15,
            'sentence_uniformity': 0.20,
            'burstiness': 0.20,
            'phrase_patterns': 0.15,
            'punctuation_regularity': 0.10,
            'starter_repetition': 0.10,
            'paragraph_structure': 0.10,
        }
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        spanish_markers = ['el', 'la', 'de', 'que', 'en', 'es', 'por', 'con', 'para', 'como']
        english_markers = ['the', 'is', 'at', 'of', 'to', 'in', 'it', 'for', 'on', 'with']
        
        words = text.lower().split()
        spanish_count = sum(1 for w in words if w in spanish_markers)
        english_count = sum(1 for w in words if w in english_markers)
        
        return 'es' if spanish_count > english_count else 'en'
    
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization."""
        text = unicodedata.normalize('NFKC', text)
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def calculate_vocabulary_diversity(self, words: List[str]) -> float:
        """
        Type-Token Ratio (TTR) - AI tends to have lower diversity.
        Human writing typically has more varied vocabulary.
        """
        if len(words) < 10:
            return 0.5
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # AI typically has TTR between 0.4-0.6
        # Human writing often has higher variation
        if ttr < 0.35:
            return 0.8  # Very repetitive = likely AI
        elif ttr < 0.45:
            return 0.65
        elif ttr < 0.55:
            return 0.5
        elif ttr < 0.65:
            return 0.35
        else:
            return 0.2  # High diversity = likely human
    
    def calculate_sentence_uniformity(self, sentences: List[str]) -> float:
        """
        AI-generated text tends to have more uniform sentence lengths.
        Human writing has higher variance.
        """
        if len(sentences) < 3:
            return 0.5
        
        lengths = [len(s.split()) for s in sentences]
        mean_length = statistics.mean(lengths)
        
        if mean_length == 0:
            return 0.5
            
        try:
            std_dev = statistics.stdev(lengths)
            cv = std_dev / mean_length  # Coefficient of variation
        except:
            return 0.5
        
        # AI typically has CV between 0.2-0.4
        # Human writing often has CV > 0.5
        if cv < 0.2:
            return 0.85  # Very uniform = likely AI
        elif cv < 0.35:
            return 0.7
        elif cv < 0.5:
            return 0.5
        elif cv < 0.7:
            return 0.3
        else:
            return 0.15  # High variance = likely human
    
    def calculate_burstiness(self, words: List[str]) -> float:
        """
        Burstiness measures how 'bursty' word usage is.
        AI tends to distribute words more evenly.
        Human writing has more clustered/bursty patterns.
        """
        if len(words) < 50:
            return 0.5
        
        # Calculate word frequency variance
        word_counts = Counter(words)
        frequencies = list(word_counts.values())
        
        if len(frequencies) < 5:
            return 0.5
        
        try:
            mean_freq = statistics.mean(frequencies)
            variance = statistics.variance(frequencies)
            burstiness = (variance - mean_freq) / (variance + mean_freq) if (variance + mean_freq) > 0 else 0
        except:
            return 0.5
        
        # Low burstiness = AI-like (even distribution)
        # High burstiness = Human-like (clustered usage)
        if burstiness < 0.1:
            return 0.75
        elif burstiness < 0.3:
            return 0.55
        elif burstiness < 0.5:
            return 0.4
        else:
            return 0.25
    
    def calculate_phrase_patterns(self, text: str, language: str) -> float:
        """
        Detect common AI phrases and patterns.
        """
        text_lower = text.lower()
        phrases = self.AI_PHRASES.get(language, self.AI_PHRASES['en'])
        
        matches = sum(1 for phrase in phrases if phrase in text_lower)
        word_count = len(text.split())
        
        # Normalize by text length
        density = (matches / (word_count / 100)) if word_count > 0 else 0
        
        if density > 3:
            return 0.9  # Many AI phrases
        elif density > 2:
            return 0.75
        elif density > 1:
            return 0.55
        elif density > 0.5:
            return 0.4
        else:
            return 0.2  # Few AI phrases
    
    def calculate_punctuation_regularity(self, text: str) -> float:
        """
        AI tends to use punctuation more regularly.
        Human writing has more varied punctuation patterns.
        """
        sentences = self.get_sentences(text)
        if len(sentences) < 3:
            return 0.5
        
        # Check comma usage per sentence
        comma_counts = [s.count(',') for s in sentences]
        
        if not comma_counts or max(comma_counts) == 0:
            return 0.5
        
        try:
            mean_commas = statistics.mean(comma_counts)
            std_commas = statistics.stdev(comma_counts) if len(comma_counts) > 1 else 0
            cv = std_commas / mean_commas if mean_commas > 0 else 0
        except:
            return 0.5
        
        # Low variance in punctuation = AI-like
        if cv < 0.3:
            return 0.7
        elif cv < 0.6:
            return 0.5
        else:
            return 0.3
    
    def calculate_starter_repetition(self, sentences: List[str], language: str) -> float:
        """
        AI often starts sentences with similar words.
        """
        if len(sentences) < 5:
            return 0.5
        
        starters = [s.split()[0] if s.split() else '' for s in sentences]
        starter_counts = Counter(starters)
        
        ai_starters = self.AI_STARTERS.get(language, self.AI_STARTERS['en'])
        ai_starter_ratio = sum(starter_counts.get(s, 0) for s in ai_starters) / len(sentences)
        
        # Check for repetitive starters
        most_common_ratio = starter_counts.most_common(1)[0][1] / len(sentences) if starter_counts else 0
        
        score = (ai_starter_ratio * 0.5) + (most_common_ratio * 0.5)
        
        if score > 0.5:
            return 0.8
        elif score > 0.35:
            return 0.6
        elif score > 0.2:
            return 0.4
        else:
            return 0.25
    
    def calculate_paragraph_structure(self, text: str) -> float:
        """
        AI tends to create very structured paragraphs.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.5
        
        para_lengths = [len(p.split()) for p in paragraphs]
        
        try:
            mean_len = statistics.mean(para_lengths)
            std_len = statistics.stdev(para_lengths) if len(para_lengths) > 1 else 0
            cv = std_len / mean_len if mean_len > 0 else 0
        except:
            return 0.5
        
        # Very uniform paragraphs = AI-like
        if cv < 0.2:
            return 0.75
        elif cv < 0.4:
            return 0.55
        else:
            return 0.35
    
    def analyze(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """
        Run full detection analysis.
        """
        import time
        start_time = time.time()
        
        # Detect language if auto
        if language == 'auto':
            language = self.detect_language(text)
        
        words = self.tokenize(text)
        sentences = self.get_sentences(text)
        
        # Calculate all signals
        signals = []
        weighted_sum = 0
        
        # 1. Vocabulary Diversity
        vocab_score = self.calculate_vocabulary_diversity(words)
        signals.append({
            'name': 'Vocabulary Diversity',
            'score': vocab_score,
            'weight': self.weights['vocabulary_diversity'],
            'description': 'Measures unique word usage. AI tends to be more repetitive.'
        })
        weighted_sum += vocab_score * self.weights['vocabulary_diversity']
        
        # 2. Sentence Uniformity
        uniform_score = self.calculate_sentence_uniformity(sentences)
        signals.append({
            'name': 'Sentence Uniformity',
            'score': uniform_score,
            'weight': self.weights['sentence_uniformity'],
            'description': 'AI produces more uniform sentence lengths.'
        })
        weighted_sum += uniform_score * self.weights['sentence_uniformity']
        
        # 3. Burstiness
        burst_score = self.calculate_burstiness(words)
        signals.append({
            'name': 'Burstiness',
            'score': burst_score,
            'weight': self.weights['burstiness'],
            'description': 'Human writing has more clustered word patterns.'
        })
        weighted_sum += burst_score * self.weights['burstiness']
        
        # 4. Phrase Patterns
        phrase_score = self.calculate_phrase_patterns(text, language)
        signals.append({
            'name': 'AI Phrase Patterns',
            'score': phrase_score,
            'weight': self.weights['phrase_patterns'],
            'description': 'Detection of common AI writing patterns.'
        })
        weighted_sum += phrase_score * self.weights['phrase_patterns']
        
        # 5. Punctuation Regularity
        punct_score = self.calculate_punctuation_regularity(text)
        signals.append({
            'name': 'Punctuation Regularity',
            'score': punct_score,
            'weight': self.weights['punctuation_regularity'],
            'description': 'AI uses punctuation more consistently.'
        })
        weighted_sum += punct_score * self.weights['punctuation_regularity']
        
        # 6. Starter Repetition
        starter_score = self.calculate_starter_repetition(sentences, language)
        signals.append({
            'name': 'Sentence Starters',
            'score': starter_score,
            'weight': self.weights['starter_repetition'],
            'description': 'AI often starts sentences with similar words.'
        })
        weighted_sum += starter_score * self.weights['starter_repetition']
        
        # 7. Paragraph Structure
        para_score = self.calculate_paragraph_structure(text)
        signals.append({
            'name': 'Paragraph Structure',
            'score': para_score,
            'weight': self.weights['paragraph_structure'],
            'description': 'AI creates more uniform paragraph lengths.'
        })
        weighted_sum += para_score * self.weights['paragraph_structure']
        
        # Calculate final probability
        ai_probability = min(max(weighted_sum, 0), 1)
        
        # Determine verdict and confidence
        if ai_probability >= 0.75:
            verdict = 'AI_GENERATED'
            confidence = 'HIGH'
        elif ai_probability >= 0.55:
            verdict = 'LIKELY_AI'
            confidence = 'MEDIUM'
        elif ai_probability >= 0.45:
            verdict = 'MIXED'
            confidence = 'LOW'
        elif ai_probability >= 0.25:
            verdict = 'LIKELY_HUMAN'
            confidence = 'MEDIUM'
        else:
            verdict = 'HUMAN_WRITTEN'
            confidence = 'HIGH'
        
        analysis_time = int((time.time() - start_time) * 1000)
        
        return {
            'ai_probability': round(ai_probability, 3),
            'confidence': confidence,
            'verdict': verdict,
            'signals': signals,
            'language_detected': language,
            'word_count': len(words),
            'character_count': len(text),
            'analysis_time_ms': analysis_time,
        }

# Initialize detection engine
detector = AIDetectionEngine()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "DETECTAR.AI",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "detect": "POST /api/v1/detect",
            "history": "GET /api/v1/history",
            "stats": "GET /api/v1/stats",
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_ai_content(request: DetectionRequest):
    """
    Analyze text content for AI generation probability.
    
    - **text**: Content to analyze (50-50,000 characters)
    - **language**: auto, en, or es
    """
    try:
        # Run detection
        result = detector.analyze(request.text, request.language)
        
        # Generate unique ID
        scan_id = hashlib.sha256(
            f"{request.text[:100]}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        timestamp = datetime.utcnow().isoformat()
        
        # Store in history
        scan_history.append({
            'id': scan_id,
            'timestamp': timestamp,
            'text_preview': request.text[:100] + '...' if len(request.text) > 100 else request.text,
            'ai_probability': result['ai_probability'],
            'verdict': result['verdict'],
            'word_count': result['word_count'],
        })
        
        # Keep only last 100 scans
        if len(scan_history) > 100:
            scan_history.pop(0)
        
        return DetectionResponse(
            id=scan_id,
            timestamp=timestamp,
            ai_probability=result['ai_probability'],
            confidence=result['confidence'],
            verdict=result['verdict'],
            signals=[DetectionSignal(**s) for s in result['signals']],
            language_detected=result['language_detected'],
            word_count=result['word_count'],
            character_count=result['character_count'],
            analysis_time_ms=result['analysis_time_ms'],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
async def get_scan_history(limit: int = 20):
    """Get recent scan history."""
    return {
        "scans": scan_history[-limit:][::-1],
        "total": len(scan_history),
    }

@app.get("/api/v1/stats")
async def get_stats():
    """Get detection statistics."""
    if not scan_history:
        return {
            "total_scans": 0,
            "ai_detected": 0,
            "human_detected": 0,
            "mixed": 0,
            "avg_ai_probability": 0,
        }
    
    ai_count = sum(1 for s in scan_history if s['verdict'] in ['AI_GENERATED', 'LIKELY_AI'])
    human_count = sum(1 for s in scan_history if s['verdict'] in ['HUMAN_WRITTEN', 'LIKELY_HUMAN'])
    mixed_count = sum(1 for s in scan_history if s['verdict'] == 'MIXED')
    avg_prob = statistics.mean(s['ai_probability'] for s in scan_history)
    
    return {
        "total_scans": len(scan_history),
        "ai_detected": ai_count,
        "human_detected": human_count,
        "mixed": mixed_count,
        "avg_ai_probability": round(avg_prob, 3),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
