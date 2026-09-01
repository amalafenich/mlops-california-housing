FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY score.py .

EXPOSE 5001

# azmlinfsrv : serveur HTTP fourni par azureml-inference-server-http,
# appelle automatiquement init() et run() définis dans score.py.
CMD ["azmlinfsrv", "--entry_script", "score.py", "--port", "5001"]
