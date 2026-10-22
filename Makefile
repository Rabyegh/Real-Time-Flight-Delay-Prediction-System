ETL_INPUT ?= data/raw
ETL_OUTPUT ?= data/processed
FEATURES_INPUT ?= data/processed/accepted
FEATURES_OUTPUT ?= data/features
TRAIN_INPUT ?= data/features/dataset
TRAIN_OUTPUT ?= models/baseline
MLFLOW_TRACKING_URI ?= file:./mlruns
MLFLOW_EXPERIMENT ?= flight-delay-baseline
KAFKA_BOOTSTRAP ?= localhost:9092
KAFKA_TOPIC ?= flight-events
KAFKA_DLQ_TOPIC ?= flight-events-dlq
STREAM_VALID_OUT ?= data/streaming/valid
STREAM_REJECTED_OUT ?= data/streaming/rejected
STREAM_CHECKPOINTS ?= data/streaming/checkpoints

.PHONY: install lint format test historical-etl features train up down logs smoke kafka-up kafka-down kafka-topics kafka-produce stream stream-clean

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

kafka-up:
	docker compose --env-file .env.example up -d kafka

kafka-down:
	docker compose --env-file .env.example stop kafka

kafka-topics:
	docker compose --env-file .env.example exec kafka kafka-topics --bootstrap-server localhost:29092 --create --if-not-exists --topic $(KAFKA_TOPIC) --partitions 3 --replication-factor 1
	docker compose --env-file .env.example exec kafka kafka-topics --bootstrap-server localhost:29092 --create --if-not-exists --topic $(KAFKA_DLQ_TOPIC) --partitions 1 --replication-factor 1

kafka-produce:
	uv run flight-delay-produce --bootstrap-servers "$(KAFKA_BOOTSTRAP)" --topic "$(KAFKA_TOPIC)" --count 100 --rate 10

stream:
	uv run flight-delay-stream --bootstrap-servers "$(KAFKA_BOOTSTRAP)" --topic "$(KAFKA_TOPIC)" --valid-output "$(STREAM_VALID_OUT)" --rejected-output "$(STREAM_REJECTED_OUT)" --checkpoint-dir "$(STREAM_CHECKPOINTS)"

stream-clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/streaming/valid', 'data/streaming/rejected', 'data/streaming/checkpoints']]"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

smoke:
	uv run python scripts/smoke_test.py
