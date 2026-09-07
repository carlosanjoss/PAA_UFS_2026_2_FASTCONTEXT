# Shared stage with the Python runtime and project dependencies.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    FASTCONTEXT_ENV=development

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# Validation stage: the image is built only when all tests pass.
FROM base AS test

RUN python -m pytest -v

# Runtime stage: keeps the application command separate from test execution.
FROM base AS runtime

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]