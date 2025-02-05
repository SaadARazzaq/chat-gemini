# Use a slim Python image as the base
ARG PYTHON_VERSION=3.10.0
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install LibreOffice and dependencies
RUN apt-get update && apt-get install -y \
    libreoffice \
    libreoffice-common \
    libreoffice-writer \
    libreoffice-core \
    libreoffice-base \
    fonts-dejavu \
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

# Change ownership of the /app directory    
RUN chown appuser:appuser /app            

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user
USER appuser

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit application
CMD streamlit run app.py
