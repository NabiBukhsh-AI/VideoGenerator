# Runtime image for videogen. ffmpeg and a caption font are baked in, which are the
# two things that most often trip up a fresh install.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIDEOGEN_OUTPUT_DIR=/data/output \
    VIDEOGEN_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

RUN apt-get update && apt-get install --no-install-recommends -y \
        ffmpeg \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first so dependency layers cache across source edits.
COPY pyproject.toml README.md ./
COPY videogen ./videogen

RUN pip install --upgrade pip && pip install ".[all]"

# Run as a non-root user; /data is the mount point for generated output.
RUN useradd --create-home --uid 1000 videogen \
    && mkdir -p /data/output \
    && chown -R videogen:videogen /data
USER videogen
VOLUME ["/data"]

EXPOSE 8501

# Default to the CLI. Override for the UI:
#   docker run -p 8501:8501 videogen ui --host 0.0.0.0
ENTRYPOINT ["videogen"]
CMD ["--help"]
