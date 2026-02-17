#!/bin/bash
# Download Essentia pre-trained models
# Models are ~87MB total

set -e

MODEL_DIR="${HOME}/essentia_models"

echo "🎵 Essentia Model Downloader"
echo "================================"
echo ""
echo "Downloading models to: ${MODEL_DIR}"
echo "Total size: ~87MB"
echo ""

# Create directory
mkdir -p "${MODEL_DIR}"
cd "${MODEL_DIR}"

# Download embedding model (required)
echo "📦 Downloading embedding model..."
wget -q --show-progress https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
wget -q --show-progress https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json

# Download genre classifier
echo "📦 Downloading genre classifier (Discogs 400)..."
wget -q --show-progress https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
wget -q --show-progress https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json

# Download mood classifier
echo "📦 Downloading mood classifier..."
wget -q --show-progress https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb
wget -q --show-progress https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json

echo ""
echo "✅ Download complete!"
echo ""
echo "Models installed:"
ls -lh "${MODEL_DIR}"

echo ""
echo "🎸 Ready to run: python tag_music.py"
