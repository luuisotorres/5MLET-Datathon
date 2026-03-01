# Data Pipeline Documentation

This project implements a **Medallion Architecture** to process student data from the Passos Mágicos Association. The pipeline is designed for reproducibility, data quality, and observability.

## 1. Architecture Overview

The data flows through four distinct stages:

1.  **Landing**: Raw Excel files provided by the NGO.
2.  **Bronze**: Raw data converted to Parquet with technical metadata.
3.  **Silver**: Cleaned, standardized data with feature engineering and drift detection.
4.  **Gold**: Aggregated historical data with time-series target creation for machine learning.

---

## 2. Layer Details

### 🟢 Landing Layer (`data/00_landing`)
- **Format**: `.xlsx` (Excel).
- **Content**: Annual student performance reports (PEDE).
- **Process**: New files are placed here manually or via upload.

> **Note on File Format**: While `.xlsx` is not the ideal format for high-scale data engineering due to schema volatility and high memory usage during parsing, it was adopted here to maintain compatibility with the NGO's current data export capabilities. The pipeline explicitly handles this by converting these to Parquet in the Bronze layer as the first step.

### 🥉 Bronze Layer (`data/01_bronze`)
- **Script**: `src/passos_magicos/data/make_bronze.py`
- **Actions**:
    - Reads Excel sheets and converts them to **Parquet** for better performance and schema preservation.
    - Adds technical metadata: `metadata_source` (filename), `metadata_sheet`, and `metadata_ingestion_date`.
    - Moves processed files from Landing to `data/archive`.

### 🥈 Silver Layer (`data/02_silver`)
- **Script**: `src/passos_magicos/data/make_silver.py`
- **Actions**:
    - **Header Standardization**: Maps heterogeneous column names (e.g., `inde_2023`, `inde_23`, `inde`) to a stable internal schema.
    - **IPP Reconstruction**: For years where the Psycho-Pedagogical Index (IPP) is missing (e.g., 2022), it is reconstructed using the weighted formula of its components (IAN, IDA, IEG, IAA, IPS, IPV).
    - **Cleaning**: 
        - `RA`: Standardized to integers.
        - `Genero`: Normalized to categorical codes.
        - `Idade`: Validated and cleaned of outliers.
        - `Pedra`: Standardized classification (Quartzo, Ágata, Ametista, Topázio).
    - **Feature Engineering**: Calculates `anos_na_instituicao` by comparing `ano_ingresso` with the current data year.
    - **Validation**: Enforces a strict data contract using **Pandera**.
    - **Drift Detection**: Uses **Evidently AI** to compare the incoming batch against the historical baseline. Reports are generated in `data/reports`.

### 🥇 Gold Layer (`data/03_gold`)
- **Script**: `src/passos_magicos/data/make_gold.py`
- **Actions**:
    - **Temporal Shifting**: Since the goal is to predict "Defasagem" (Academic Lagging), the pipeline shifts the target variable from $Year_{N+1}$ to $Year_{N}$.
    - **Offline Store**: Saves `training_data.parquet` containing only student transitions (rows where both current features and next-year target exist).
    - **Online Store**: Syncs the latest available features for every student to an **SQLite** database (`feature_store_online.db`) for low-latency API inference.

---

## 3. Data Governance & Maintenance

- **Schema Enforcement**: Any breaking change in the source Excel files will be caught by the Silver layer's Pandera validation.
- **Drift Reports**: Always check `data/reports/drift_report_*.html` after running the pipeline to ensure the new data distribution matches the training baseline.
- **Cleaning Environment**: Use `make clean` to wipe all layers and restart the ingestion from Landing.

---

## 4. Recommendations for Future Data Ingestion

To improve scalability and data reliability, future iterations should transition toward more robust data delivery methods:

1. **Standardized Serialization**: Prioritize **CSV** or **Parquet** at the source. These formats are less prone to metadata corruption compared to Excel and are significantly faster to process.
2. **Schema-First Exports**: Ensure the source system provides consistent column naming conventions across years to minimize the overhead of mapping heterogeneous headers in the Silver layer.
3. **Direct Database Integration**: Instead of manual file uploads to a Landing folder, implement a direct extraction from the NGO's management system (ERP/CRM) into a **Cloud Data Lake**.
4. **Data Validation at Source**: Implement automated validation checks before the data is landed to ensure mandatory fields (like Student RA) are present and correctly formatted.
