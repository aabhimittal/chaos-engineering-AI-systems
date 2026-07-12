.PHONY: help install dev test lint demo serve docker-build compose-up compose-down clean

help:
	@echo "chaoslab — AI-native Chaos Lab"
	@echo ""
	@echo "  make install       install the package (offline core only)"
	@echo "  make dev           install with dev + all optional backends"
	@echo "  make test          run the pytest suite"
	@echo "  make lint          run ruff"
	@echo "  make demo          run every bundled chaos experiment"
	@echo "  make serve         run the chaos agent locally (/metrics on :8000)"
	@echo "  make docker-build  build the chaos-agent image"
	@echo "  make compose-up    bring up the full local stack (prom/grafana/mlflow)"
	@echo "  make compose-down  tear the stack down"
	@echo "  make clean         remove caches and local run store"

install:
	pip install -e .

dev:
	pip install -e ".[all,dev]"

test:
	pytest

lint:
	ruff check chaoslab tests

demo:
	chaoslab run experiments/

serve:
	chaoslab serve --dir experiments --interval 60 --port 8000

docker-build:
	docker build -t chaoslab:latest .

compose-up:
	docker compose -f deploy/docker-compose.yml up -d --build

compose-down:
	docker compose -f deploy/docker-compose.yml down -v

clean:
	rm -rf .chaoslab .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist
