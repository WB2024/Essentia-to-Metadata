FROM python:3.10-slim

# Install system dependencies required by Essentia
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libfftw3-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libsamplerate0-dev \
        libtag1-dev \
        libchromaprint-dev \
        libyaml-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
# essentia-tensorflow only publishes dev builds for Python 3.10+,
# which are not matched by the >=2.1b6 specifier in requirements.txt,
# so we install it explicitly first then the remaining deps.
RUN uv pip install --system --no-cache essentia-tensorflow
COPY requirements.txt .
RUN uv pip install --system --no-cache mutagen numpy

# Download ML models at build time
COPY download_models.sh .
ENV HOME=/app
RUN sed -i 's/\r$//' download_models.sh && bash download_models.sh

# Copy application code
COPY tag_music.py docker-entrypoint.py ./

# Suppress noisy TensorFlow logs (INFO + WARNING)
ENV TF_CPP_MIN_LOG_LEVEL=3

# Music directory mount point
VOLUME ["/music"]

ENTRYPOINT ["python", "docker-entrypoint.py", "--model-dir", "/app/essentia_models"]
CMD ["/music", "--auto"]
