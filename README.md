# EC Price Predictor

Predicts Executive Condominium (EC) resale prices in Singapore at two key milestones:
- **5-year MOP** (Minimum Occupancy Period)
- **10-year Privatisation**

Built with **LightGBM** trained on URA private residential property transaction data, served via a **FastAPI** REST service, containerised with **Docker**, and deployed on **Railway.app**.

## Quick start (local Docker)

```bash
cp .env.example .env          # URA key is pre-filled
docker compose up -d --build
# Wait ~5 min for auto-ingest + training, then:
open http://localhost:8000/ui   # Frontend
open http://localhost:8000/docs # API docs
```

## Make commands

| Command | What it does |
|---|---|
| `make up` | Build + start all services |
| `make logs` | Tail API logs |
| `make ingest` | Trigger URA data pull |
| `make train` | Trigger model training |
| `make predict` | Run a sample prediction |
| `make health` | Check service health |

## Docs

See [`docs/SOLUTION.md`](docs/SOLUTION.md) for full documentation including:
- Data model ER diagram
- ML approach & model selection rationale
- API design
- Cloud architecture
- Model monitoring, automation & governance proposals

## API

```
POST /predict          ← Main endpoint
GET  /health
GET  /ui               ← Frontend
GET  /docs             ← Swagger
POST /admin/ingest     ← Requires X-Admin-Token header
POST /admin/train      ← Requires X-Admin-Token header
```
