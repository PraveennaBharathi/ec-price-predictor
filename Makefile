.PHONY: up down build logs ingest train test

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f api

ingest:
	curl -s -X POST http://localhost:8000/admin/ingest?background=false \
	  -H "X-Admin-Token: $$(grep ADMIN_TOKEN .env | cut -d= -f2)" | python3 -m json.tool

train:
	curl -s -X POST "http://localhost:8000/admin/train?horizon=0&background=false" \
	  -H "X-Admin-Token: $$(grep ADMIN_TOKEN .env | cut -d= -f2)" | python3 -m json.tool

predict:
	curl -s -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"area_sqft":1076,"floor_level":8,"district":19,"market_segment":"OCR","lease_commencement_year":2019}' \
	  | python3 -m json.tool

health:
	curl -s http://localhost:8000/health | python3 -m json.tool
