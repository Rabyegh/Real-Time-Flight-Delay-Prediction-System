ETL_INPUT ?= data/raw
ETL_OUTPUT ?= data/processed
FEATURES_INPUT ?= data/processed/accepted
FEATURES_OUTPUT ?= data/features
TRAIN_INPUT ?= data/features/dataset
TRAIN_OUTPUT ?= models/baseline
MLFLOW_TRACKING_URI ?= file:./mlruns
MLFLOW_EXPERIMENT ?= flight-delay-baseline

.PHONY: install lint format test historical-etl features train up down logs smoke

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run python -m pytest

historical-etl:
	uv run flight-delay-etl --input "$(ETL_INPUT)" --output "$(ETL_OUTPUT)"

features:
	uv run flight-delay-features --input "$(FEATURES_INPUT)" --output "$(FEATURES_OUTPUT)"

train:
	uv run flight-delay-train --input "$(TRAIN_INPUT)" --output "$(TRAIN_OUTPUT)" --tracking-uri "$(MLFLOW_TRACKING_URI)" --experiment-name "$(MLFLOW_EXPERIMENT)"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

smoke:
	uv run python scripts/smoke_test.py
