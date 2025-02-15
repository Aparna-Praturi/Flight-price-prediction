FROM apache/airflow:latest

USER root

# Install system dependencies and build tools
RUN apt-get update && \
    apt-get install -y git gcc python3-dev build-essential && \
    apt-get clean

# Copy requirements and fix permissions
COPY requirements.txt /opt/airflow/
#RUN chown airflow:root /opt/airflow/requirements.txt

# Switch to airflow user for pip installations
USER airflow
RUN  pip install --no-cache-dir --prefer-binary joblib scikit-learn matplotlib







