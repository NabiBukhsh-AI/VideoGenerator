.PHONY: help install dev test test-fast lint format typecheck check clean demo ui

help:
	@echo "install    Install the package with all extras"
	@echo "dev        Install with dev tooling too"
	@echo "test       Run the full test suite"
	@echo "test-fast  Skip tests that need ffmpeg"
	@echo "lint       Run ruff"
	@echo "format     Auto-format with ruff"
	@echo "typecheck  Run mypy"
	@echo "check      lint + typecheck + test"
	@echo "demo       Render a sample video offline (no API keys needed)"
	@echo "ui         Launch the Streamlit interface"
	@echo "clean      Remove build and cache artifacts"

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[all,dev]"

test:
	pytest

test-fast:
	pytest -m "not ffmpeg"

lint:
	ruff check videogen tests

format:
	ruff format videogen tests
	ruff check --fix videogen tests

typecheck:
	mypy videogen

check: lint typecheck test

demo:
	python -m videogen generate --dry-run -o output/demo \
		"The deep ocean remains the least explored region on Earth. Sunlight fades \
		entirely below one thousand metres, leaving a permanent night. Creatures there \
		produce their own light through bioluminescence."

ui:
	python -m videogen ui

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
