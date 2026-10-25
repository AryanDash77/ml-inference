FROM python:3.11-slim

WORKDIR /app

# XGBoost and LightGBM need the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK data for the sentiment project, downloaded once at build time
# (the running pod has no internet access). The build fails if any download fails.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import nltk, sys; sys.exit(0 if all(nltk.download(p, download_dir='/usr/share/nltk_data') for p in ('stopwords', 'wordnet', 'punkt_tab')) else 1)"

# Copy app code and model
COPY app/ ./app/
COPY models/ ./models/

ENV MODEL_DIR=/app/models
EXPOSE 8000

# Basic container-level healthcheck (separate from the k8s probes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

