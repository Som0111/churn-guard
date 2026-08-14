FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependencies first so the layer caches across code changes.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY README.md ./
RUN pip install --no-cache-dir -e .

# Bake the trained model into the image so the container starts ready to serve.
RUN python -m churnguard.train --skip-figures

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
