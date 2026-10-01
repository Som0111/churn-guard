.PHONY: help install data train test serve lint clean all

help:
	@echo "install  - install the package and dev dependencies"
	@echo "data     - download the raw dataset"
	@echo "train    - run the full training pipeline"
	@echo "test     - run the test suite"
	@echo "serve    - start the prediction API on :8000"
	@echo "lint     - run ruff"
	@echo "all      - install, train, test"

install:
	pip install -c constraints.txt -e ".[dev]"

data:
	python -c "from churnguard import data; data.download()"

train:
	python -m churnguard.train

test:
	pytest --cov=churnguard --cov-report=term-missing

serve:
	uvicorn churnguard.api:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff check src tests

clean:
	rm -rf models/*.joblib reports/figures/*.png reports/metrics.json .pytest_cache

all: install train test
