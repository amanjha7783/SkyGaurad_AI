# SkyGuard AI — Modified Real-Observation Dataset

## What was changed
The supplied dataset was inspected and standardized without changing the underlying real observations in `data/real/real_aws_baseline.csv`.

The supplied file contains 1,922 observations from 7 stations, covering 2025-01-01 00:00:00 to 2025-01-06 23:30:00. The timestamps are predominantly 30-minute observations.

## Real baseline
Source metadata supplied with the dataset identifies the source as:
NOAA / NWS ASOS Network via IEM Archive.

The baseline files preserve the supplied measurements. Missing pressure/RH values remain missing.

## Controlled anomaly dataset
`data/ml/anomaly_injected_aws.csv` is a COPY of the baseline with controlled faults injected for supervised ground-truth evaluation.

Injected examples:
- SPIKE
- FROZEN_SENSOR
- DRIFT
- OFFSET
- MISSING_DATA
- COMMUNICATION_ERROR
- PRESSURE_ANOMALY
- HUMIDITY_SENSOR_FAILURE
- TEMPORAL_ANOMALY
- MULTI_SENSOR_FAILURE
- INTERMITTENT_FAILURE

No injected fault is applied to `data/real/real_aws_baseline.csv`.

## Important limitation
The supplied dataset is NOT six months. It covers only 2025-01-01 through 2025-01-06. It should therefore be described as a real-data pilot/sample, not as six months of observations.

To create a true six-month real-data version, additional real source records must be supplied/downloaded. Do not duplicate or interpolate these rows and call the result real.

## Files
- data/real/real_aws_baseline.csv
- data/real/clean_aws.csv
- data/ml/anomaly_injected_aws.csv
- data/ml/train.csv
- data/ml/validation.csv
- data/ml/test.csv
- metadata/source_metadata.json
- metadata/anomaly_config.json
- reports/data_quality_report.csv
- reports/dataset_statistics.csv

## Recommended project wording
"SkyGuard AI uses real weather-station observations as the baseline and injects controlled sensor faults only into a copy of the observations to create ground-truth labels for supervised evaluation."

## Disclaimer
The observations are represented according to the source metadata supplied with the uploaded dataset. The anomaly labels are synthetic and were created for SkyGuard AI evaluation.
