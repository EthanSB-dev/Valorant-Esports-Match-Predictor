FROM apache/airflow:2.9.3

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    "dbt-core==1.8.2" \
    "dbt-postgres==1.8.2" \
    "pandas==2.2.2" \
    "scikit-learn==1.5.1" \
    "joblib==1.4.2"