.PHONY: lint tests

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run isort equiv/ tests/

tests:
	uv run pytest tests/ --cov=equiv --cov-report=term-missing
