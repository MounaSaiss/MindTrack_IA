
from datetime import datetime, timedelta                            #   datetime:définir date/heure , timedelta:définir durée(m/h/j)

from airflow import DAG
from airflow.operators.bash import BashOperator #execute cmd dans terminal de conteneur 


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="student_mental_health_pipeline",
    default_args=default_args,
    description="Pipeline ML for Student Mental Health Risk Prediction",
    start_date=datetime(2026, 11, 12),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["machine-learning", "mental-health"],
) as dag:

    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command="python /opt/airflow/src/data_preparation.py",
    )

    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command="python /opt/airflow/src/preprocessing.py",
    )

    training = BashOperator(
        task_id="training",
        bash_command="python /opt/airflow/src/train.py",
    )

    evaluation = BashOperator(
        task_id="evaluation",
        bash_command="python /opt/airflow/src/evaluate.py",
    )

    data_preparation >> preprocessing >> training >> evaluation
