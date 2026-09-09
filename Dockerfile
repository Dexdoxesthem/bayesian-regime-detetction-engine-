FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for numpy/scipy/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend build stage
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Final image
FROM base AS final
WORKDIR /app

COPY src/ src/
COPY api/ api/
COPY data/ data/
COPY r/ r/
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# Precompute all API responses so the service responds instantly on boot
# (trains HMM + Deep Ensemble once here, bakes results into the image)
RUN python api/precompute.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
