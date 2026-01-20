# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install build dependencies
RUN pip install --no-cache-dir build==1.3.0 poetry

# Install poetry plugin
RUN poetry self add poetry-plugin-export

# Set the working directory
WORKDIR /app

# Copy the project files
COPY pyproject.toml poetry.lock ./
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Generate requirements.txt from poetry.lock
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Build the wheel
RUN python -m build --wheel --outdir /wheels


# Stage 2: Runtime
FROM python:3.12-slim AS runtime

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

# Install dependencies from requirements.txt to ensure lockfile is respected
RUN pip install --no-cache-dir -r requirements.txt

# Install the application wheel
RUN pip install --no-cache-dir /wheels/*.whl
