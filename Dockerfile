FROM python:3.13-slim

WORKDIR /app

#uv for dependency installation
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

COPY stage_1/ stage_1/
COPY stage_2/ stage_2/

EXPOSE 8080
CMD ["uvicorn", "api:app", "--app-dir", "stage_2", "--host", "0.0.0.0", "--port", "8080"]