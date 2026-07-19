.PHONY: install verify backend frontend test lint build compose-up compose-down

install:
	python -m pip install -r backend/requirements.txt
	cd frontend && npm ci

verify:
	python -m compileall -q backend inference dev/agentic
	python -m pytest backend/tests -q
	cd frontend && npm run lint && npx tsc --noEmit && npm run build

backend:
	python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

test:
	python -m pytest backend/tests -q

lint:
	python -m ruff check backend inference dev/agentic dev/rag/lightweight_retriever.py
	cd frontend && npm run lint

build:
	cd frontend && npm run build

compose-up:
	docker compose up --build

compose-down:
	docker compose down
