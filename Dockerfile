FROM python:3.11-slim

LABEL org.opencontainers.image.title="chaoslab" \
      org.opencontainers.image.description="AI-native Chaos Lab — chaos engineering for AI/LLM systems"

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY chaoslab ./chaoslab
RUN pip install --no-cache-dir ".[prometheus,mlflow]"

# Default experiments (can be overridden by a mounted ConfigMap volume).
COPY experiments ./experiments

# Non-root user for safety.
RUN useradd --create-home chaos && chown -R chaos /app
USER chaos

EXPOSE 8000

# Run the chaos agent: scrape experiments, expose /metrics, loop on an interval.
# Kubernetes overrides --dir to the ConfigMap mount (/etc/chaoslab/experiments).
ENTRYPOINT ["chaoslab", "serve"]
CMD ["--dir", "experiments", "--port", "8000", "--interval", "300"]
