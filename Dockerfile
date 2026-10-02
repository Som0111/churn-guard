# Production Python is 3.12 (same as Render and the README).
# Pinned by digest as well as tag, so a rebuild cannot silently pick up a different base.
FROM python:3.12.7-slim-bookworm@sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Commit this image was built from (no .git inside the build context).
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

# pyproject.toml is the single source of truth; constraints.txt pins every resolved version.
COPY pyproject.toml constraints.txt README.md ./
COPY src/ ./src/

# EXTRAS=explain adds SHAP drivers to /predict (~150 MB of numba/llvmlite).
# Build with --build-arg EXTRAS= for a slim image that scores without drivers.
# Editable, so the package (and the models/ and reports/ it writes) live under /app.
ARG EXTRAS=explain
RUN pip install -c constraints.txt -e ".${EXTRAS:+[$EXTRAS]}"

# Bake the trained model into the image so the container starts ready to serve.
RUN python -m churnguard.train --skip-figures

# Run as an unprivileged user: it can read the model but owns nothing it could tamper with.
RUN useradd --system --no-create-home --uid 10001 churnguard
USER churnguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/ready').status==200 else 1)"

CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
