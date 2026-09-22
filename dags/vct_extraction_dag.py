"""
VCT extraction + load + transform DAG.

Orchestrates the full extract -> load -> transform flow:
1. Pull new flagship VCT matches from PandaScore (extract/extract_matches.py)
2. Load the resulting raw JSON files into the raw_matches Postgres table
   (load/load_matches.py), using an ELT pattern.
3. Run dbt to transform raw_matches into staging/intermediate/marts models
   (transform/vct_dbt), producing fct_matches, dim_teams, dim_tournaments.

Credentials for both PandaScore and Postgres are resolved from Airflow
Connections / environment variables at runtime, never hardcoded in this file.
"""
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.hooks.base import BaseHook

sys.path.insert(0, "/opt/airflow/extract")
sys.path.insert(0, "/opt/airflow/load")

DBT_PROJECT_DIR = "/opt/airflow/transform/vct_dbt"

def run_load(**context):
    import load_matches

    conn = BaseHook.get_connection("vct_warehouse")
    conn_uri = conn.get_uri()

    load_matches.load_all_raw_matches(conn_uri=conn_uri)

    context["ti"].xcom_push(key="load_completed_at", value=datetime.utcnow().isoformat())

def run_extraction(**context):
    from pandascore_client import PandaScoreClient
    import extract_matches

    conn = BaseHook.get_connection("pandascore_api")
    api_key = conn.password

    original_client_cls = extract_matches.PandaScoreClient
    extract_matches.PandaScoreClient = lambda: PandaScoreClient(api_key=api_key)

    try:
        extract_matches.extract_new_matches()
    finally:
        extract_matches.PandaScoreClient = original_client_cls

    context["ti"].xcom_push(key="run_completed_at", value=datetime.utcnow().isoformat())


def run_extraction_upcoming(**context):
    from pandascore_client import PandaScoreClient
    import extract_upcoming_matches

    conn = BaseHook.get_connection("pandascore_api")
    api_key = conn.password

    original_client_cls = extract_upcoming_matches.PandaScoreClient
    extract_upcoming_matches.PandaScoreClient = lambda: PandaScoreClient(api_key=api_key)

    try:
        extract_upcoming_matches.extract_upcoming_matches()
    finally:
        extract_upcoming_matches.PandaScoreClient = original_client_cls


def run_prediction(**context):
    sys.path.insert(0, "/opt/airflow/analysis")
    from sqlalchemy import create_engine
    import predict_upcoming

    conn = BaseHook.get_connection("vct_warehouse")
    conn_uri = conn.get_uri()
    # SQLAlchemy needs the explicit psycopg2 dialect in the URL scheme;
    # Airflow connection URIs don't include it.
    if conn_uri.startswith("postgresql://"):
        conn_uri = conn_uri.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif conn_uri.startswith("postgres://"):
        conn_uri = conn_uri.replace("postgres://", "postgresql+psycopg2://", 1)

    engine = create_engine(conn_uri)
    predict_upcoming.main(engine=engine)

def log_summary(**context):
    extracted_at = context["ti"].xcom_pull(key="run_completed_at", task_ids="run_extraction")
    loaded_at = context["ti"].xcom_pull(key="load_completed_at", task_ids="run_load")
    print(f"Extraction finished at: {extracted_at}")
    print(f"Load finished at: {loaded_at}")
    print("Transform (dbt run + dbt test) completed successfully.")


with DAG(
    dag_id="vct_extraction",
    description="Extract new flagship VCT matches, load into Postgres, and transform with dbt",
    start_date=datetime(2026, 8, 1),
    schedule="0 */6 * * *",
    catchup=False,
    tags=["vct", "extraction", "load", "transform"],
) as dag:

    extract_task = PythonOperator(
        task_id="run_extraction",
        python_callable=run_extraction,
    )

    extract_upcoming_task = PythonOperator(
        task_id="run_extraction_upcoming",
        python_callable=run_extraction_upcoming,
    )

    load_task = PythonOperator(
        task_id="run_load",
        python_callable=run_load,
    )

    dbt_run_task = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt run",
    )

    dbt_test_task = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt test",
    )

    predict_task = PythonOperator(
        task_id="run_prediction",
        python_callable=run_prediction,
    )

    summary_task = PythonOperator(
        task_id="log_summary",
        python_callable=log_summary,
    )

    [extract_task, extract_upcoming_task] >> load_task >> dbt_run_task >> dbt_test_task >> predict_task >> summary_task