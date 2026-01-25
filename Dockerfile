# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install build dependencies
# hadolint ignore=DL3013
RUN pip install --no-cache-dir build==1.3.0 poetry && \
    poetry self add poetry-plugin-export

# Set the working directory
WORKDIR /app

# Copy the project files
COPY pyproject.toml poetry.lock ./
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Generate requirements.txt from poetry.lock and build the wheel
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    python -m build --wheel --outdir /wheels


# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Install system dependencies
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Set the working directory
WORKDIR /home/appuser/app

# Copy requirements and wheel from the builder stage
COPY --from=builder /app/requirements.txt .
COPY --from=builder /wheels /wheels

# Install dependencies from requirements.txt and the application wheel
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir /wheels/*.whl

CMD ["uvicorn", "coreason_refinery.server:app", "--host", "0.0.0.0", "--port", "8000"]
