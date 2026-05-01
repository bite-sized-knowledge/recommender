FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# uv: statically linked binary, build-essential 불필요 (numpy/polars wheel 사용)
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

WORKDIR /app

# 의존성만 먼저 (lock 변경 시만 cache invalidate)
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev

COPY . /app

CMD ["/app/.venv/bin/python", "main.py"]
