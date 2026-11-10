# Agulhas transport pipeline, built for the machine it runs on: x86-64, or ARM such as Apple silicon.
#   docker build -t agulhas .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/figures:/app/figures" agulhas
# On x86-64 every pinned package has a prebuilt wheel. On ARM, cartopy, numcodecs and rasterio
# have none at the pinned versions, so the build stage compiles them once, rasterio against
# Debian's GDAL; the pins themselves are the same on both.

FROM python:3.9-slim AS build
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends build-essential libgdal-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt pytest

FROM python:3.9-slim
ARG TARGETARCH
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# rasterio's x86-64 wheel bundles GDAL but links against the system expat, which the slim image
# leaves out; on ARM it was built against Debian's GDAL, which it needs at run time
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
        $(if [ "$TARGETARCH" = "arm64" ]; then echo libgdal36; fi) \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl

WORKDIR /app
COPY pyproject.toml ./
COPY agulhas/ agulhas/
COPY tests/ tests/
COPY docs/ docs/

CMD ["python", "-m", "agulhas", "all"]
