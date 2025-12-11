# DETECTAR.AI — AI Content Detection Platform

Enterprise-grade AI detection API with multi-signal analysis for Spanish and English content. EU AI Act compliant.

## Quick Start

### 1. Start the API Server

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Open the Dashboard

Open `frontend/index.html` in your browser.

### 3. Test via API

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content to analyze here...", "language": "auto"}'
```

## Detection Signals

The engine analyzes 7 statistical signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Vocabulary Diversity | 15% | Type-Token Ratio - AI tends to be repetitive |
| Sentence Uniformity | 20% | AI produces uniform sentence lengths |
| Burstiness | 20% | Human writing has clustered word patterns |
| AI Phrase Patterns | 15% | Common AI writing phrases detected |
| Punctuation Regularity | 10% | AI uses punctuation consistently |
| Sentence Starters | 10% | AI starts sentences with similar words |
| Paragraph Structure | 10% | AI creates uniform paragraph lengths |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check |
| POST | `/api/v1/detect` | Analyze text for AI probability |
| GET | `/api/v1/history` | Recent scan history |
| GET | `/api/v1/stats` | Detection statistics |

## Response Format

```json
{
  "id": "abc123...",
  "timestamp": "2024-12-12T10:30:00Z",
  "ai_probability": 0.65,
  "confidence": "MEDIUM",
  "verdict": "LIKELY_AI",
  "signals": [...],
  "language_detected": "en",
  "word_count": 150,
  "character_count": 850,
  "analysis_time_ms": 45
}
```

## Verdicts

| Probability | Verdict | Confidence |
|-------------|---------|------------|
| ≥75% | AI_GENERATED | HIGH |
| 55-74% | LIKELY_AI | MEDIUM |
| 45-54% | MIXED | LOW |
| 25-44% | LIKELY_HUMAN | MEDIUM |
| <25% | HUMAN_WRITTEN | HIGH |

## Docker Deployment

```bash
docker-compose up -d
```

- API: http://localhost:8000
- Dashboard: http://localhost:3000

## Pricing Tiers

| Plan | Price | API Calls | Features |
|------|-------|-----------|----------|
| Starter | €99/mo | 10,000 | Email support, Basic analytics |
| Professional | €299/mo | 100,000 | Priority support, Webhooks |
| Enterprise | Custom | Unlimited | On-premise, SLA |

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Vanilla JS, HTML5, CSS3
- **Detection**: Statistical analysis engine (no ML dependencies)

## License

Proprietary - AGENTS AI Limited © 2024
