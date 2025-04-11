# Microserviço IA Trade (FastAPI + CLIP)

## Como rodar com Docker

```bash
docker build -t ia-trade-service .
docker run -p 8001:8001 ia-trade-service
```

## Como rodar com VS Code

1. Certifique-se que você tem o Python 3.10+ e `pip` instalados.
2. Rode:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Endpoints

- `POST /teach` → Envia imagem + explicação
- `POST /analyze` → Envia imagem para comparação

As informações são armazenadas em `memory.json` no diretório raiz.