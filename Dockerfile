# Use a slim Python image as the base
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /app

# Install necessary dependencies including LibreOffice
RUN apt-get update && apt-get install --yes --no-install-recommends \
    libreoffice \
    libreoffice-common \
    libreoffice-writer \
    libreoffice-core \
    libreoffice-base \
    fonts-dejavu \
    gcc \
    g++ \
    build-essential \
    software-properties-common \
    git \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify that soffice exists
RUN which soffice || ls /usr/bin/soffice || exit 1

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Change ownership of /app directory
RUN chown appuser:appuser /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy any additional system packages (if needed)
COPY packages.txt packages.txt
RUN xargs -a packages.txt apt-get install --yes || true

# Switch to the non-privileged user
USER appuser

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "app.py"]
