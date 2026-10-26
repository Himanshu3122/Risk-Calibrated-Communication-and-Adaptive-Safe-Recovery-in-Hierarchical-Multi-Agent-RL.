# CPU image for running tests and small experiments. For GPU training, swap the torch wheel index.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r requirements-dev.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini ./

RUN useradd --create-home ricor && chown -R ricor /app
USER ricor

CMD ["python", "scripts/smoke_test.py"]