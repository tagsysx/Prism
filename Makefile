# Makefile for Prism: Wideband RF Neural Radiance Fields
# Common development tasks and project management

.PHONY: help install install-dev install-docs clean test lint format docs build dist clean-build clean-pyc clean-test clean-docs

# Default target
help:
	@echo "Prism: Wideband RF Neural Radiance Fields"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install the package in development mode"
	@echo "  install-dev  - Install development dependencies"
	@echo "  install-docs - Install documentation dependencies"
	@echo "  test         - Run tests with pytest"
	@echo "  lint         - Run linting with flake8"
	@echo "  format       - Format code with black"
	@echo "  docs         - Build documentation"
	@echo "  build        - Build distribution packages"
	@echo "  clean        - Clean all build artifacts"
	@echo "  clean-build  - Clean build artifacts"
	@echo "  clean-pyc    - Clean Python cache files"
	@echo "  clean-test   - Clean test artifacts"
	@echo "  clean-docs   - Clean documentation build"
	@echo ""

# Installation
install:
	pip install -e .

install-dev: install
	pip install -e ".[dev]"

install-docs: install
	pip install -e ".[docs]"

# Testing
test:
	python -m pytest tests/ -v --cov=src/prism --cov-report=term-missing --cov-report=html

test-fast:
	python -m pytest tests/ -v

# Code quality
lint:
	flake8 src/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/ --line-length=88
	isort src/ tests/ scripts/

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Building
build:
	python -m build

dist: build

# Cleaning
clean: clean-build clean-pyc clean-test clean-docs

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-pyc:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

clean-docs:
	rm -rf docs/_build/

# Development setup
setup-dev: install-dev
	pre-commit install

# Quick development cycle
dev-cycle: format lint test

# Docker (if needed)
docker-build:
	docker build -t prism-rf .

docker-run:
	docker run -it --gpus all -v $(PWD):/workspace prism-rf

# Jupyter notebook setup
notebook-setup: install
	pip install -e ".[notebooks]"
	python -m ipykernel install --user --name=prism-rf --display-name="Prism RF"

# Environment management
venv:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

venv-activate:
	@echo "To activate virtual environment, run: source venv/bin/activate"

# Data management
data-download:
	@echo "Downloading sample datasets..."
	# Add data download commands here

data-clean:
	@echo "Cleaning datasets..."
	# Add data cleaning commands here

# Model training examples
train-wifi:
	python scripts/prism_runner.py --mode train --config configs/ofdm-wifi.yml --dataset_type ofdm --gpu 0

train-wideband:
	python scripts/prism_runner.py --mode train --config configs/ofdm-wideband.yml --dataset_type ofdm --gpu 0

train-5g:
	python scripts/prism_runner.py --mode train --config configs/ofdm-5g-sionna.yml --dataset_type ofdm --gpu 0

# Model testing examples
test-wifi:
	python scripts/prism_runner.py --mode test --config configs/ofdm-wifi.yml --dataset_type ofdm --gpu 0 --checkpoint checkpoints/ofdm_wifi/best_model.pth

test-wideband:
	python scripts/prism_runner.py --mode test --config configs/ofdm-wideband.yml --dataset_type ofdm --gpu 0 --checkpoint checkpoints/ofdm_1024/best_model.pth

test-5g:
	python scripts/prism_runner.py --mode test --config configs/ofdm-5g-sionna.yml --dataset_type ofdm --gpu 0 --checkpoint checkpoints/ofdm_5g/best_model.pth

# Demo
demo:
	python scripts/basic_usage.py

# Performance profiling
profile:
	python -m cProfile -o profile.stats scripts/basic_usage.py
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Security checks
security-check:
	safety check
	bandit -r src/ -f json -o bandit-report.json

# Dependency management
update-deps:
	pip install --upgrade -r requirements.txt
	pip freeze > requirements.txt

# Git hooks
git-hooks: install-dev
	pre-commit install --hook-type pre-commit
	pre-commit install --hook-type commit-msg

# CI/CD helpers
ci-install:
	pip install -e ".[dev,test]"

ci-test:
	python -m pytest tests/ --cov=src/prism --cov-report=xml --cov-report=term-missing

ci-lint:
	flake8 src/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports

ci-format-check:
	black --check src/ tests/ scripts/ --line-length=88
	isort --check-only src/ tests/ scripts/
