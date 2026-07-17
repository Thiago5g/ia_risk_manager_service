# ia_risk_manager_service

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

AI-powered trade chart analysis microservice using vision models and pattern memory.

## Architecture

```
┌────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────┐
│ Chart Image│────▶│ CLIP Embeddings  │────▶│ Pattern Matching │────▶│ GPT Analysis │────▶│ Response │
└────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘     └──────────┘
                                                     ▲
                                                     │
                                              ┌──────────────┐
                                              │ memory.json  │
                                              └──────────────┘
```

## How It Works

1. **Teach** — Upload a trade chart image with an explanation. CLIP generates an embedding, which is stored alongside your annotation in `memory.json`.
2. **Store** — Learned patterns persist in a lightweight JSON file (no database required).
3. **Analyze** — Upload a new chart image. The service computes its CLIP embedding, finds the closest learned patterns, and sends context to GPT for a detailed analysis and recommendation.

## Quick Start

### Docker (recommended)

```bash
# Clone and configure
git clone https://github.com/Thiago5g/ia_risk_manager_service.git
cd ia_risk_manager_service
cp .env.example .env  # Add your OPENAI_API_KEY

# Run
docker-compose up --build
```

### Local

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY=your_key_here
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Reference

### POST /teach

Upload a chart image and teach the system a new pattern.

| Parameter     | Type   | Description                        |
|---------------|--------|------------------------------------|
| `image`       | file   | Trade chart image (PNG/JPG)        |
| `explanation` | string | Description of the pattern/setup   |

```bash
curl -X POST http://localhost:8000/teach \
  -F "image=@chart.png" \
  -F "explanation=Bullish engulfing at support level"
```

### POST /analyze

Upload a chart image for analysis against learned patterns.

| Parameter | Type | Description                 |
|-----------|------|-----------------------------|
| `image`   | file | Trade chart image (PNG/JPG) |

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@new_chart.png"
```

Returns a JSON response with pattern matches and GPT-generated analysis.

## Tech Stack

| Component       | Technology              |
|-----------------|------------------------|
| Runtime         | Python 3.10+           |
| Framework       | FastAPI                |
| Vision Model    | CLIP (OpenAI)          |
| Language Model  | OpenAI GPT             |
| Storage         | memory.json (file)     |
| Containerization| Docker                 |
| Deployment      | Fly.io                 |

## Environment Variables

| Variable         | Required | Description           |
|------------------|----------|-----------------------|
| `OPENAI_API_KEY` | Yes      | OpenAI API key        |

## Deployment

Production deployment is configured via `fly.toml` for [Fly.io](https://fly.io):

```bash
fly deploy
```

## License

MIT
