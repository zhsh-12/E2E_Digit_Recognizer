.PHONY: install backend frontend dev test coverage docker-up docker-down clean

# ============================================================
# Digit Recognizer App — Common Make Commands
# ============================================================

## Install Python dependencies (requires uv)
install:
	uv sync

## Start backend (FastAPI dev server)
backend:
	uv run uvicorn api.main:app --reload --port 8000

## Start frontend (Vite dev server)
frontend:
	cd frontend && npm run dev

## Start both backend and frontend (requires two terminals)
dev:
	@echo "Run 'make backend' and 'make frontend' in separate terminals."
	@echo "Or use Docker: make docker-up"

## Run all tests with coverage report
test:
	uv run pytest

## Open coverage HTML report
coverage:
	uv run pytest --cov --cov-report=html
	@echo "Coverage report generated at htmlcov/index.html"

## Start with Docker Compose
docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

## Stop Docker services
docker-down:
	docker compose -f docker/docker-compose.yml down

## Clean up Python/frontend caches and build artifacts
clean:
	rm -rf .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned."
