FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    BAYWIND_CACHE_DIR=/tmp/baywind-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /tmp/baywind-cache
EXPOSE 10000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 150 --access-logfile - --error-logfile - app:app"]
