# Agulhas transport pipeline. Build and run for x86-64, where every pinned package has a wheel:
#   docker build --platform linux/amd64 -t agulhas .
#   docker run --rm --platform linux/amd64 -v "$PWD/data:/app/data" -v "$PWD/figures:/app/figures" agulhas
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# rasterio's wheel links against the system expat, which the slim image leaves out
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY pyproject.toml ./
COPY agulhas/ agulhas/
COPY tests/ tests/
COPY docs/ docs/

CMD ["python", "-m", "agulhas", "all"]
