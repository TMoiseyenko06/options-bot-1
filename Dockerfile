FROM python:3.11-slim

# System deps for XGBoost (CUDA-free) + SSL tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip certifi urllib3 \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code (data/ and models/ are mounted at runtime)
COPY backtest/       backtest/
COPY dashboard/      dashboard/
COPY data_pipeline/  data_pipeline/
COPY features/       features/
COPY grid_search/    grid_search/
COPY models/         models/
COPY run_pipeline.sh .

# Placeholder dirs — real data comes from volume mounts
RUN mkdir -p data/raw data/processed data/dbn data/cache \
             backtest/results grid_search/results

# Use up-to-date CA bundle from certifi
ENV SSL_CERT_FILE=/usr/local/lib/python3.11/site-packages/certifi/cacert.pem
ENV REQUESTS_CA_BUNDLE=/usr/local/lib/python3.11/site-packages/certifi/cacert.pem

CMD ["python", "backtest/engine.py"]
