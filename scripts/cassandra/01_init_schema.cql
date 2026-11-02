-- Phase 6: Cassandra Schema for Real-Time Flight Delay Prediction Pipeline
-- Keyspace: flight_delay (Replication factor 1 for local development)

CREATE KEYSPACE IF NOT EXISTS flight_delay
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

-- Table 1: predictions_by_event
-- Query Pattern: Retrieve prediction by deterministic event ID (GET /predictions/{event_id})
-- Partition Key: event_id
-- Idempotency: Repeated writes for the same event ID upsert deterministically.
CREATE TABLE IF NOT EXISTS flight_delay.predictions_by_event (
    event_id text,
    carrier text,
    flight_number text,
    origin text,
    dest text,
    event_time timestamp,
    prediction_time timestamp,
    prediction_date date,
    delay_probability double,
    predicted_delayed boolean,
    model_version text,
    PRIMARY KEY (event_id)
);

-- Table 2: predictions_by_airport
-- Query Pattern: Retrieve recent predictions for an airport (GET /airports/{airport_code}/predictions)
-- Partition Key: (airport, prediction_date)
--   Time-bucketing by prediction date bounds partition size to a single day (~2,500 rows / 250KB for large hubs),
--   avoiding unbounded partition growth and performance degradation.
-- Clustering Keys: prediction_time DESC, event_id ASC
--   Pre-sorts predictions chronologically in reverse order so latest predictions are read first.
--   Secondary clustering on event_id guarantees deterministic identity even if timestamps collide.
CREATE TABLE IF NOT EXISTS flight_delay.predictions_by_airport (
    airport text,
    prediction_date date,
    prediction_time timestamp,
    event_id text,
    carrier text,
    flight_number text,
    dest text,
    event_time timestamp,
    delay_probability double,
    predicted_delayed boolean,
    model_version text,
    PRIMARY KEY ((airport, prediction_date), prediction_time, event_id)
) WITH CLUSTERING ORDER BY (prediction_time DESC, event_id ASC);
