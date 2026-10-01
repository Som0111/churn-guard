# Production Python is 3.12 (same as Render and the README).
FROM python:3.12.7-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1

# Commit this image was built from (no .git inside the build context).
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

# Dependencies first so the layer caches across code changes. pyproject.toml is
# the single source of truth; constraints.txt pins every resolved version.
COPY pyproject.toml constraints.txt ./
COPY src/ ./src/
COPY README.md ./
# EXTRAS=explain adds SHAP drivers to /predict (~150 MB of numba/llvmlite).
# Build with --build-arg EXTRAS= for a slim image that scores without drivers.
ARG EXTRAS=explain
RUN pip install -c constraints.txt ".${EXTRAS:+[$EXTRAS]}"

# Bake the trained model into the image so the container starts ready to serve.
RUN python -m churnguard.train --skip-figures

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s     CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
