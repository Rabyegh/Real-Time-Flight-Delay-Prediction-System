FROM python:3.11.11-slim

RUN pip install --no-cache-dir "mlflow==2.22.5" "boto3==1.43.116"

EXPOSE 5000
