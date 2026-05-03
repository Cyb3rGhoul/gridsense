# Data Sources

## Downloaded Public Sources

- `data/raw/sgcc`: public SGCC electricity theft detection repository with split archives `data.zip`, `data.z01`, and `data.z02`.
- `data/raw/sgcc_extracted/data.csv`: extracted SGCC labelled theft dataset used by the theft-validation lab.
- `data/raw/lead`: public LEAD repository, including `data/lead1.0-small.zip`.
- `data/raw/kaggle/india`: real CEEW Indian smart-meter data for Bareilly and Mathura, downloaded via Kaggle.
- `data/raw/kaggle/london`: real London smart-meter dataset, downloaded via Kaggle.

## Active Real Dataset

- `data/raw/kaggle/london/hhblock_dataset/hhblock_dataset/block_0.csv`: active operational dashboard source. It contains real half-hourly household smart-meter readings.
- `data/raw/kaggle/london/weather_hourly_darksky.csv`: real weather source joined into the operational forecasting model.
- `data/raw/kaggle/india/SM Cleaned Data BR2019.csv`: India-domain reference source. It contains 3-minute readings with kWh, voltage, current, frequency, and meter IDs.
- `data/raw/sgcc_extracted/data.csv`: active labelled theft-validation source.
- `data/raw/lead/data/lead1.0-small.zip`: available labelled anomaly source with 200 building meters, hourly readings across 2016, and anomaly labels.

## Generated Output Files

- `data/processed/sample_meter_readings.csv`: sampled real smart-meter readings after utility-style field mapping.
- `data/processed/forecasts.json`: 24-hour feeder-level forecast output.
- `data/processed/anomalies.json`: explainable anomaly queue.
- `data/processed/anomaly_evidence.json`: evidence traces for dashboard anomaly details.
- `data/processed/zones.json`: locality/feeder risk summary.
- `data/processed/pipeline_summary.json`: model/pipeline stages shown in the dashboard.
- `data/processed/theft_validation.json`: SGCC labelled theft-validation metrics, feature importance, and top cases.
- `data/processed/metrics.json`: model and data summary metrics.

## Credential-Gated Sources

Kaggle datasets listed in the proposal require a Kaggle API token and accepted competition/dataset rules. They should be used for extended training, but they are intentionally not required for the judge demo.
