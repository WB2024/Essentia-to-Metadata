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

# Install Python dependencies
# essentia-tensorflow only publishes dev builds for Python 3.10+,
# which are not matched by the >=2.1b6 specifier in requirements.txt,
# so we install it explicitly first then the remaining deps.
RUN pip install --no-cache-dir essentia-tensorflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ML models at build time
COPY download_models.sh .
ENV HOME=/app
RUN bash download_models.sh

# Copy application code
COPY tag_music.py .

# Default model directory inside the container
ENV MODEL_DIR=/app/essentia_models

# Music directory mount point
VOLUME ["/music"]

ENTRYPOINT ["python", "tag_music.py"]
CMD ["/music", "--auto"]
