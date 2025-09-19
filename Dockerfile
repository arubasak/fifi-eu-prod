# --- Stage 1: Builder ---
# This stage installs dependencies and compiles any code needed.
# It will be discarded later to keep the final image small.
FROM python:3.12-slim as builder

WORKDIR /app

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies first to leverage Docker's layer caching.
# This step is only re-run if requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Image ---
# This is the final, clean image that will be deployed.
FROM python:3.12-slim as final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install Nginx (but not build-essential, which we no longer need)
RUN apt-get update && apt-get install -y --no-install-recommends nginx && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the application code and necessary config files
COPY fifi.py .
COPY production_config.py .
COPY fingerprint_component.html .
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh .

# Make the startup script executable
RUN chmod +x ./start.sh

# Expose the port Nginx will listen on
EXPOSE $PORT

# Run the startup script
CMD ["./start.sh"]

