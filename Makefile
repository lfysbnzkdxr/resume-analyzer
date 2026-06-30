.PHONY: install run test lint format typecheck clean

install:
	pip install -e ".[dev]"

run:
	streamlit run src/main.py

test:
	pytest -v --tb=short --cov=src

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf data/uploads/* data/chroma_db/* .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
