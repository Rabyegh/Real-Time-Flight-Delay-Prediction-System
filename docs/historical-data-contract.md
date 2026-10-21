# Historical flight CSV profile and contract

## Inspected files

Both files were read as complete CSV streams on 2026-10-21. They are readable
as UTF-8 with optional byte-order-mark handling (`utf-8-sig`), and the detected
delimiter is a comma. CSV quoting correctly preserves commas inside city names.
The CSV parser found no rows with a field count different from the header.

| Local file | Rows | Columns | Flight-date range | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `data/raw/feb.csv` | 502,749 | 22 | 2023-02-01 to 2023-02-28 | `224650ea47c7e7f0d067fed073d94910bd7b85ff1b1ecf470e9644e322c66bb3` |
| `data/raw/june.csv` | 577,262 | 22 | 2023-06-01 to 2023-06-30 | `6e434700eb0696a51a9f9410a27a2f9cf988be77e1b7df78a148af107ab5e39c` |

The files have exactly the same ordered header. All non-empty values in the
numeric-looking columns parsed as numbers during the streaming profile. No
whitespace-trimmed, byte-for-byte-equivalent full data rows were found. The
date in `FL_DATE` agreed with `YEAR`, `MONTH`, and `DAY_OF_MONTH` for every row;
`DAY_OF_WEEK` also agreed with a Monday=1 through Sunday=7 convention.

## Source schema contract

The names below are the actual headers. Source field meanings are inferred from
those names and observed values; no accompanying BTS data dictionary was
provided in the repository. The pipeline must not infer cancellation or an
individual flight number where no such source field exists.

| Source field(s) | Observed type / proposed Spark type | Null observations (Feb / Jun) | Contract notes |
| --- | --- | ---: | --- |
| `YEAR`, `MONTH`, `DAY_OF_MONTH`, `DAY_OF_WEEK` | Integer / `IntegerType` | 0 / 0 each | Date components; check consistency with parsed `FL_DATE`. Observed ranges: year 2023, month 2 or 6, day 1–28 or 1–30, weekday 1–7. |
| `FL_DATE` | Text in `M/d/yyyy h:mm:ss a` / `DateType` | 0 / 0 | The flight-date field. Parse with the explicit observed format and validate against date components. |
| `OP_UNIQUE_CARRIER`, `ORIGIN`, `ORIGIN_CITY_NAME`, `ORIGIN_STATE_NM`, `DEST`, `DEST_CITY_NAME`, `DEST_STATE_NM` | Text / `StringType` | 0 / 0 each | Carrier, origin, destination, and display-name fields as supplied. Preserve as strings. |
| `CRS_DEP_TIME`, `CRS_ARR_TIME` | Numeric HHmm / `IntegerType` | 0 / 0 each | Scheduled time fields. Values range from 1 to 2359/2400. Keep the source HHmm representation as an integer; do not interpret it as elapsed minutes. |
| `DEP_TIME` | Numeric HHmm / nullable `IntegerType` | 8,815 / 11,603 | Actual departure-time field. Preserve missing as null. Observed maximum is 2400. |
| `DEP_DELAY` | Numeric decimal text / nullable `DoubleType` | 8,815 / 11,607 | Field whose name indicates departure delay. Preserve missing values; never replace with zero. Observed values are integer-valued but represented as decimal strings. |
| `DEP_DEL15` | Numeric indicator / nullable `IntegerType` | 8,815 / 11,607 | Observed non-null values are 0 or 1. Preserve null; validate the observed binary domain. Confirm the precise threshold semantics from the source dictionary before relying on it as a model label. |
| `ARR_DELAY` | Numeric decimal text / nullable `DoubleType` | 10,002 / 14,458 | Field whose name indicates arrival delay. Preserve missing values; never replace with zero. Observed values are integer-valued but represented as decimal strings. |
| `DEP_TIME_BLK`, `ARR_TIME_BLK` | Text / `StringType` | 0 / 0 each | Supplied time-block labels; keep as text. |
| `CRS_ELAPSED_TIME`, `DISTANCE` | Numeric decimal text / nullable `DoubleType` | 1 / 0; 0 / 0 | Source numeric measures. Values observed are positive; do not impute the one missing elapsed-time value. Confirm units from BTS source metadata before attaching units to downstream fields. |

There is no `CANCELLED`, `CANCELLATION_CODE`, `FL_NUM`, or other explicit
flight-number column. In particular, missing actual times or delays are **not**
proof of cancellation.

## Profile observations and quality constraints

| Field | February observed values | June observed values |
| --- | --- | --- |
| `DEP_TIME` | 1–2400 | 1–2400 |
| `DEP_DELAY` | -45–2947 | -48–3695 |
| `DEP_DEL15` | 0–1 | 0–1 |
| `CRS_ARR_TIME` | 1–2359 | 1–2400 |
| `ARR_DELAY` | -86–2949 | -119–3680 |
| `CRS_ELAPSED_TIME` | 24–685 | 17–674 |
| `DISTANCE` | 31–5095 | 31–5095 |

No non-numeric values were found in non-empty numeric fields, no invalid clock
values were found in the three observed HHmm fields, all non-null
`DEP_DEL15` values were binary, and all non-null elapsed-time and distance
values were positive. Large positive delay values are present in the source;
without an authoritative source rule they are not to be capped or rejected as
outliers.

Missing values are limited to the fields and counts listed above. In February,
the missing counts for `DEP_TIME`, `DEP_DELAY`, and `DEP_DEL15` differ by four
between `DEP_TIME` and the latter two. That discrepancy is retained and reported;
the pipeline must not infer values from neighboring fields.

There are no exact full-row duplicates in either file. A candidate tuple of
`FL_DATE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, and `CRS_DEP_TIME` repeats in
136 February rows and 178 June rows. Because no flight number is present, that
tuple is not a reliable unique key. The ETL may report exact duplicate
candidates, but must not delete records based on this composite tuple or claim
that it uniquely identifies a flight.

## Ingestion and normalization rules

1. Read CSV using UTF-8 with BOM support and header-based parsing; do not infer
   schema from file order alone.
2. Normalize header names only by removing a leading BOM, trimming surrounding
   whitespace, and converting to uppercase. Reject duplicate names created by
   normalization and report missing, unexpected, or incompatible fields.
3. Require the canonical 22 source fields listed above. A schema change is an
   explicit issue, not a reason to silently discard or coerce columns.
4. Parse `FL_DATE` with the observed month/day/year and 12-hour-time format,
   then validate it against the integer date components.
5. Parse numeric fields explicitly. Blank or malformed optional measurements
   remain null or are rejected with a recorded field-specific reason; unknown
   delays are never converted to zero.
6. Validate date components, weekday, clock encodings (including observed
   `2400`), binary `DEP_DEL15`, and positive non-null elapsed time/distance.
   Do not reject long delay values solely for being unusually large.
7. There is no defensible flight-level deduplication key in these extracts.
   Preserve rows, report exact duplicate candidates, and do not silently
   drop records based on the proposed composite tuple.

The raw files are local inputs and are excluded by `.gitignore`; neither these
files nor generated Parquet outputs belong in Git.
