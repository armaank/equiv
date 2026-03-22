.PHONY: lint tests clean figures

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run isort equiv/ tests/

tests:
	uv run pytest tests/ --cov=equiv --cov-report=term-missing

figures:
	uv run python equiv/main.py

clean:
	rm -f coverage.xml .coverage *.png
	rm -rf .pytest_cache htmlcov
