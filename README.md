# [![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/succinctrecords)

# 🎸 Essentia Music Tagger

**Intelligent audio analysis and automatic genre/mood tagging for your music library**

Analyze your music collection using machine learning and write accurate genre and mood tags directly to your audio files. No internet required - all processing happens locally using [Essentia](https://essentia.upf.edu/), an advanced audio analysis library from the Music Technology Group.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Essentia](https://img.shields.io/badge/Essentia-2.1b6-green.svg)](https://essentia.upf.edu/)

---

## ✨ Features

- 🎯 **Audio-based analysis** - Analyzes actual audio content, not metadata lookups
- 🎼 **400 genre classifications** - Uses Discogs taxonomy for detailed genre tagging
- 😊 **Mood detection** - Detects moods like energetic, dark, happy, aggressive, etc.
- 🔄 **Batch processing** - Recursively process entire music libraries
- 🎛️ **Fully configurable** - Interactive prompts for all settings on each run
- 📝 **Comprehensive logging** - Detailed logs with confidence scores and predictions
- 🏷️ **Multiple tag formats** - Choose how genre tags are formatted
- 💾 **FLAC & MP3 support** - Writes to standard tag formats
- 🧪 **Dry run mode** - Test before making changes
- 🚀 **CPU-only** - No GPU required (though it helps!)

---

## 🎵 How It Works

Unlike tools like MusicBrainz Picard or beets that look up metadata from online databases, **Essentia Music Tagger analyzes the actual audio waveform** using deep learning models:

1. **Loads audio** - Reads your music files
2. **Extracts features** - Analyzes spectral, tonal, and rhythmic characteristics
3. **Runs ML models** - Pre-trained neural networks predict genres and moods
4. **Writes tags** - Saves predictions to your music files

**No internet connection required after initial setup!**

### Example Predictions

**Input:** Alternative rock track  
**Output:**

🎸 Genres: Rock - Alternative Rock (32%), Rock - Indie Rock (23%), Rock - Brit Pop (22%) 😊 Moods: Energetic (2.3%), Dark (1.8%)

Tags written:

- `GENRE`: `Rock - Alternative Rock; Rock - Indie Rock; Rock - Brit Pop`
- `MOOD`: `Energetic; Dark`

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Linux** (Debian/Ubuntu recommended, also works on macOS)
- **~100MB disk space** for models
- **8GB+ RAM** recommended

### Installation

```bash
# 1. Clone the repository
cd /srv/dev-disk-by-uuid-dc4918d5-6597-465b-9567-ce442fbd8e2a/Github
git clone https://github.com/WB2024/Essentia-to-Metadata.git
cd Essentia-to-Metadata

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install essentia-tensorflow mutagen numpy

# 4. Download ML models (~87MB)
bash download_models.sh
```

### First Run

# Run the script

python tag_music.py

You'll be prompted for:

- 📂 **Music directory** to analyze
- 🎸 **Number of genres** to tag (1-10)
- 📊 **Confidence thresholds** for genre/mood
- 🎨 **Tag formatting** style
- 🧪 **Dry run mode** (test first!)
- And more...

**Recommendation:** Run in dry-run mode first to preview results!

---

## 📖 Detailed Usage

### Interactive Configuration

Every run prompts you to configure:

#### Genre Settings

- **Number of genres** (1-10) - How many genre tags per song
- **Confidence threshold** (1-50%) - Minimum prediction confidence
- **Format style**:
  - `Rock - Alternative Rock` (parent - child) ← default
  - `Alternative Rock - Rock` (child - parent)
  - `Alternative Rock` (child only)
  - `Rock---Alternative Rock` (raw)

#### Mood Settings

- **Enable/disable** mood analysis
- **Confidence threshold** (0.1-20%) - Moods have lower confidence than genres

#### Other Options

- **Dry run mode** - Test without writing tags
- **Confidence tags** - Write detailed scores to custom tags
- **Overwrite existing** - Skip or replace existing genre tags
- **Verbose output** - Show detailed predictions

### Example Session

$ python tag_music.py

🎸 ESSENTIA MUSIC TAGGER - INTERACTIVE MODE
══════════════════════════════════════════════════════════════════════

Enter the path to analyze: /music/2Pac

📂 Directory: /music/2Pac
🎵 Found ~45 audio files

⚙️  CONFIGURATION
──────────────────────────────────────────────────────────────────────
🧪 DRY RUN MODE
Enable dry run mode? [Y/n]: y

🎸 GENRE SETTINGS
Number of genres to write [3]: 3
Genre threshold (%) [15]: 15
Genre format [1]: 1

😊 MOOD ANALYSIS
Enable mood analysis? [Y/n]: y
Mood threshold (%) [0.5]: 0.5

🔄 Loading models...
✅ Models loaded successfully!

[1/45] 2Pac/Me Against the World/05 - Temptations.flac
     🎸 Raw: Hip-Hop---Gangsta (45.2%), Hip-Hop---East Coast Hip Hop (32.4%)
     🎸 Formatted: Hip-Hop - Gangsta, Hip-Hop - East Coast Hip Hop
     😊 Raw: energetic (2.3%), dark (1.8%)
     😊 Formatted: Energetic, Dark
     [DRY RUN] Would write: Genres: Hip-Hop - Gangsta, Hip-Hop - East Coast Hip Hop | Moods: Energetic, Dark

## 🎛️ Configuration Guide

### Confidence Thresholds

Understanding confidence scores:

**Genres:**

- Model predicts across **400 possible classes**
- Top prediction typically 15-40%
- **15% threshold** = balanced (recommended)
- **25% threshold** = strict (fewer genres)
- **5% threshold** = inclusive (more genres)

**Moods:**

- Naturally **much lower** than genres (0.1-5% range)
- **0.5% threshold** = good starting point
- **1-3%** = more selective

### Tag Formatting Examples

| Raw Prediction            | parent_child              | child_parent              | child_only         |
| ------------------------- | ------------------------- | ------------------------- | ------------------ |
| `Rock---Alternative Rock` | `Rock - Alternative Rock` | `Alternative Rock - Rock` | `Alternative Rock` |
| `Hip-Hop---Gangsta`       | `Hip-Hop - Gangsta`       | `Gangsta - Hip-Hop`       | `Gangsta`          |
| `Electronic---Techno`     | `Electronic - Techno`     | `Techno - Electronic`     | `Techno`           |

---

## 📁 Output

### Tags Written

**FLAC files:**

- `GENRE` - Formatted genre tags (semicolon-separated)
- `MOOD` - Formatted mood tags (semicolon-separated)
- `ESSENTIA_GENRE` - Raw predictions with confidence scores (optional)
- `ESSENTIA_MOOD` - Raw mood predictions with scores (optional)

**MP3 files:**

- `TCON` (Genre) - Formatted genre tags
- `COMM` (Comment) - Confidence scores (optional)

### Log Files

Each run creates a timestamped log: `essentia_tagger_YYYYMMDD_HHMMSS.log`

Example log content:

FILE: 2Pac/Me Against the World/05 - Temptations.flac
────────────────────────────────────────────────────────────────────────────
GENRES (raw predictions):
  • Hip-Hop---Gangsta: 45.23%
  • Hip-Hop---East Coast Hip Hop: 32.45%
  • Hip-Hop---Golden Age Hip Hop: 21.34%

GENRES (formatted for tags):
  • Hip-Hop - Gangsta
  • Hip-Hop - East Coast Hip Hop

ALL GENRE PREDICTIONS (top 10):
  • Hip-Hop---Gangsta: 45.23%
  • Hip-Hop---East Coast Hip Hop: 32.45%
  ...

MOODS (passed threshold - 2 total):
  • energetic: 2.34%
  • dark: 1.87%

## 🎓 Understanding the Models

### Genre Model: Discogs-400

- **Classes:** 400 genre/style categories from Discogs taxonomy
- **Architecture:** EfficientNet-based CNN
- **Training:** Supervised learning on Discogs-tagged releases
- **Strengths:** Very detailed genre classification
- **Example classes:** `Hip-Hop---Golden Age Hip Hop`, `Rock---Shoegaze`, `Electronic---Deep House`

### Mood Model: MTG-Jamendo

- **Classes:** Mood and theme tags (energetic, dark, happy, sad, etc.)
- **Architecture:** Multi-label classification
- **Training:** MTG-Jamendo dataset with crowd-sourced tags
- **Note:** Lower confidence than genres (this is normal!)

### Embedding Model: Discogs-Effnet

- **Purpose:** Extracts audio features for downstream tasks
- **Input:** 16kHz audio resampled from any format
- **Output:** High-dimensional embedding vectors

---

## ⚡ Performance

**Processing Speed** (CPU-only, Intel i3-4150T @ 3.00GHz):

- **~5-15 seconds** per track (varies by length and complexity)
- **~500 tracks** = 2-4 hours
- **~2000 tracks** = 8-16 hours

**Tips for faster processing:**

- Use newer/faster CPU
- Process in batches by artist/album
- Run overnight for large libraries
- GPU support (if available) can 10x speed

**Memory Usage:**

- **~2-3GB RAM** during processing
- Models loaded once, reused for all files

---

## 🛠️ Troubleshooting

### "Could not load model" errors

- Ensure models are downloaded: `bash download_models.sh`
- Check `~/essentia_models/` contains `.pb` and `.json` files

### "No moods above threshold"

- Moods have very low confidence (0.1-5%)
- Try lowering mood threshold to 0.3% or 0.1%
- Check log file for raw mood predictions

### TensorFlow warnings

Could not load dynamic library 'libcudart.so.11.0'

- **Safe to ignore** - means no GPU, will use CPU
- Performance is still good on CPU

### Out of memory

- Reduce batch processing
- Close other applications
- Upgrade RAM if processing very large files

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for contributions:

- Support for more audio formats (ALAC, Opus, etc.)
- GUI interface
- Progress bars
- Resume/checkpoint system for interrupted runs
- Custom model support
- Genre mapping/translation tables

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/copilot/c/LICENSE) file for details.

**Note:** Essentia library itself is licensed under AGPL-3.0. Pre-trained models from MTG are available under CC BY-NC-ND 4.0 for non-commercial use.

---

## 🙏 Credits

### Built With

- **[Essentia](https://essentia.upf.edu/)** - Audio analysis library by Music Technology Group
- **[Mutagen](https://mutagen.readthedocs.io/)** - Python library for audio metadata
- **[NumPy](https://numpy.org/)** - Numerical computing

### Models

Pre-trained models provided by:

- **[Music Technology Group (MTG)](https://www.upf.edu/web/mtg)** - Universitat Pompeu Fabra
- Trained on public datasets: Discogs, MTG-Jamendo, AcousticBrainz

### Inspiration

- [AcousticBrainz](https://acousticbrainz.org/) - Crowdsourced acoustic analysis
- [beets](https://beets.io/) - Music library management
- [MusicBrainz Picard](https://picard.musicbrainz.org/) - Music tagger

---

## 📚 Further Reading

- [Essentia Documentation](https://essentia.upf.edu/documentation.html)
- [MTG Models Repository](https://essentia.upf.edu/models.html)
- [Discogs Genre Taxonomy](https://www.discogs.com/help/doc/submission-guidelines-release-genres-styles)
- [Music Information Retrieval](https://musicinformationretrieval.com/)

---

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/WB2024/Essentia-to-Metadata/issues)
- **Discussions:** [GitHub Discussions](https://github.com/WB2024/Essentia-to-Metadata/discussions)

---

## 🎶 Example Use Cases

### 1. Organize Your Music Library

Tag your entire collection with accurate genres for better browsing in music players.

### 2. DJ / Producer Workflow

Quickly find tracks by mood/energy for mixing and production.

### 3. Music Research

Analyze genre distributions in music collections.

### 4. Playlist Generation

Use mood tags to create dynamic playlists (energetic workout mixes, relaxing evening music, etc.).

---

**Made with ❤️ for music lovers and data nerds**

*If this project helps you, consider ⭐ starring the repo!*
