# ia_risk_manager_service

Trade chart pattern recognition microservice using CLIP embeddings and cosine similarity.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-Vision-412991?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

## How It Works

This is a **retrieval-based** pattern recognition system, not a generative AI:

```
┌────────────┐     ┌──────────────────┐     ┌────────────────┐
│ Chart Image│────▶│ CLIP Embeddings  │────▶│ Store/Compare  │
└────────────┘     │ (512-dim vector) │     │ (memory.json)  │
                   └──────────────────┘     └───────┬────────┘
                                                    │
                                            ┌───────▼────────┐
                                            │ Cosine Similar. │
                                            │ → Best Match    │
                                            └───────┬────────┘
                                                    │
                                            ┌───────▼────────┐
                                            │ Return stored   │
                                            │ explanation     │
                                            └────────────────┘
```

**Two phases:**

1. **Teach** — Upload chart images with expert annotations. CLIP generates a 512-dimensional embedding for each image, stored alongside the annotation in `memory.json`.

2. **Analyze** — Upload a new chart. CLIP embeds it, cosine similarity finds the closest stored pattern, and returns the expert's original annotation with a confidence score.

> **Note:** This system does NOT use GPT or any language model for analysis. It's pure embedding similarity — matching visual patterns to pre-stored expert knowledge.

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Thiago5g/ia_risk_manager_service.git
cd ia_risk_manager_service
docker-compose up --build
```

### Local

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
OpenAPI docs at `http://localhost:8000/docs`.

## API Reference

### GET /health

Returns service status and pattern count.

```bash
curl http://localhost:8000/health
# {"status": "ok", "patterns_count": 5}
```

### POST /teach

Upload a chart image and teach the system a new pattern.

| Parameter     | Type   | Description                           |
|---------------|--------|---------------------------------------|
| `file`        | file   | Trade chart image (PNG/JPEG/WebP)     |
| `explanation` | string | Expert annotation of the pattern      |

```bash
curl -X POST http://localhost:8000/teach \
  -F "file=@chart.png" \
  -F "explanation=Bullish engulfing at key support level"
# {"message": "Pattern stored successfully", "patterns_stored": 6}
```

### POST /analyze

Upload a chart image for analysis against stored patterns.

| Parameter | Type | Description                    |
|-----------|------|--------------------------------|
| `file`    | file | Trade chart image (PNG/JPEG/WebP) |

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@new_chart.png"
# {"similarity": 0.87, "matched_explanation": "Bullish engulfing at key support level", "confidence": "high"}
```

**Confidence levels:**
- `high`: similarity ≥ 0.85
- `medium`: similarity ≥ 0.70
- `low`: similarity < 0.70

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Vision Model | CLIP (openai/clip-vit-base-patch32) via HuggingFace |
| Embeddings | 512-dimensional vectors |
| Similarity | Cosine similarity (numpy) |
| Storage | JSON file (memory.json) |
| Deployment | Docker, Fly.io |
| Validation | Pydantic schemas |

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEMORY_PATH` | No | `./memory.json` | Path to pattern storage file |

**No external API keys required.** CLIP runs locally using HuggingFace transformers.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests mock the CLIP model — no model download required for running tests.

## Limitations

- **Storage:** Patterns stored in a JSON file (not suitable for large-scale production)
- **No generation:** Returns stored explanations, does not generate new analysis
- **No concurrency:** File-based storage has no locking mechanism
- **Single model:** Uses CLIP ViT-B/32 only, no model selection

## Future Improvements

- [ ] Add GPT analysis on top of pattern retrieval (generate insights, not just match)
- [ ] Replace JSON storage with vector database (pgvector, Qdrant)
- [ ] Add support for multiple CLIP model variants
- [ ] Implement pattern versioning and deletion
- [ ] Add batch analysis endpoint

## License

MIT
