# Essentia Music Tagger - Picard Automation Setup Guide

This guide explains how to automatically tag your music files with Essentia genre/mood analysis whenever MusicBrainz Picard saves files to your music library.

## Overview

The setup works as follows:
1. **Picard** (in Docker) saves/moves files to your music directory
2. **inotifywait** (file system watcher) detects the new files
3. **tag_music.py** analyzes and adds genre/mood tags to the files

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  MusicBrainz    │     │   File Watcher   │     │    Essentia     │
│    Picard       │────▶│  (inotifywait)   │────▶│    Tagger       │
│   (Docker)      │     │   (systemd)      │     │  (tag_music.py) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                        │
        │                        │                        │
        ▼                        ▼                        ▼
    Saves files          Detects new files        Adds genre/mood
    to /storage/         in watch directory       tags to files
```

## Prerequisites

- OpenMediaVault 7 (OMV7) server
- MusicBrainz Picard running in Docker (already set up per your docker-compose)
- SSH access to your OMV7 server
- Root access on the server

## Step-by-Step Setup

### Step 1: Install System Dependencies

SSH into your OMV7 server and install required packages:

```bash
# Update package lists
apt update

# Install inotify-tools (for file watching)
apt install -y inotify-tools

# Install Python 3 and venv (should already be installed)
apt install -y python3 python3-pip python3-venv

# Install build dependencies for essentia
apt install -y build-essential libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libsamplerate0-dev libtag1-dev libchromaprint-dev libyaml-dev
```

### Step 2: Create the Tagger Directory

```bash
# Create the installation directory
mkdir -p /opt/essentia-tagger

# Create directories for models and logs
mkdir -p /opt/essentia-tagger/models
mkdir -p /var/log/essentia-tagger
```

### Step 3: Copy the Scripts

You need to copy three files to your OMV7 server:
- `tag_music.py` - The main tagger script
- `essentia_watcher.sh` - The file watcher script
- `essentia-tagger.service` - The systemd service file

**Option A: Using SCP from your local machine:**

```bash
# From your local Windows machine (adjust paths as needed)
scp tag_music.py root@YOUR_OMV_IP:/opt/essentia-tagger/
scp essentia_watcher.sh root@YOUR_OMV_IP:/opt/essentia-tagger/
scp essentia-tagger.service root@YOUR_OMV_IP:/etc/systemd/system/
```

**Option B: Using wget if hosted on GitHub:**

```bash
# If you push to GitHub, you can wget the raw files
cd /opt/essentia-tagger
wget https://raw.githubusercontent.com/YOUR_USER/Essentia-to-Metadata/main/tag_music.py
wget https://raw.githubusercontent.com/YOUR_USER/Essentia-to-Metadata/main/essentia_watcher.sh
wget https://raw.githubusercontent.com/YOUR_USER/Essentia-to-Metadata/main/essentia-tagger.service -O /etc/systemd/system/essentia-tagger.service
```

**Option C: Create files directly on the server:**

```bash
# SSH into server, then use nano/vim to create files
cd /opt/essentia-tagger
nano tag_music.py  # paste content
nano essentia_watcher.sh  # paste content
```

Make the watcher script executable:

```bash
chmod +x /opt/essentia-tagger/essentia_watcher.sh
```

### Step 4: Create Python Virtual Environment

```bash
cd /opt/essentia-tagger

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install numpy
pip install essentia-tensorflow
pip install mutagen

# Verify installation
python -c "from essentia.standard import MonoLoader; print('Essentia OK')"

# Deactivate when done
deactivate
```

> **Note:** If `essentia-tensorflow` fails to install, try:
> ```bash
> pip install essentia
> pip install tensorflow
> ```

### Step 5: Download Essentia Models

```bash
cd /opt/essentia-tagger/models

# Download the embedding model
wget https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb

# Download genre model and metadata
wget https://essentia.upf.edu/models/music-style-classification/discogs400/genre_discogs400-discogs-effnet-1.pb
wget https://essentia.upf.edu/models/music-style-classification/discogs400/genre_discogs400-discogs-effnet-1.json

# Download mood model and metadata
wget https://essentia.upf.edu/models/music-style-classification/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb
wget https://essentia.upf.edu/models/music-style-classification/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json

# Verify all files exist
ls -la
```

You should see these 5 files:
- `discogs-effnet-bs64-1.pb`
- `genre_discogs400-discogs-effnet-1.pb`
- `genre_discogs400-discogs-effnet-1.json`
- `mtg_jamendo_moodtheme-discogs-effnet-1.pb`
- `mtg_jamendo_moodtheme-discogs-effnet-1.json`

### Step 6: Configure the Service

Edit the systemd service file to match your paths:

```bash
nano /etc/systemd/system/essentia-tagger.service
```

**Key settings to verify/change:**

```ini
# Your watch directory (where Picard saves files)
Environment="WATCH_DIR=/srv/dev-disk-by-uuid-dc4918d5-6597-465b-9567-ce442fbd8e2a/Media/Audio/Music/Sources/Clean"

# Paths to scripts and models
Environment="TAGGER_SCRIPT=/opt/essentia-tagger/tag_music.py"
Environment="VENV_PATH=/opt/essentia-tagger/venv"
Environment="MODEL_DIR=/opt/essentia-tagger/models"
Environment="LOG_DIR=/var/log/essentia-tagger"

# Tagging settings
Environment="GENRES=3"              # Number of genres to tag
Environment="GENRE_THRESHOLD=15"    # Genre confidence % threshold
Environment="MOOD_THRESHOLD=0.5"    # Mood confidence % threshold
Environment="GENRE_FORMAT=parent_child"  # Format style

# Processing options
Environment="DRY_RUN=false"         # Set to 'true' to test without writing
Environment="OVERWRITE=true"        # Overwrite existing genre tags
Environment="DEBOUNCE_SECONDS=5"    # Wait time before processing
```

### Step 7: Test the Setup

Before enabling the service, test everything manually:

```bash
# Test 1: Check dependencies
/opt/essentia-tagger/essentia_watcher.sh --check

# Test 2: Process a single test file manually
source /opt/essentia-tagger/venv/bin/activate
python /opt/essentia-tagger/tag_music.py "/path/to/test/song.flac" \
    --auto \
    --single-file \
    --model-dir /opt/essentia-tagger/models \
    --dry-run
deactivate

# Test 3: Run the watcher in test mode (processes up to 5 existing files)
/opt/essentia-tagger/essentia_watcher.sh --test
```

### Step 8: Enable and Start the Service

```bash
# Reload systemd to pick up the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable essentia-tagger.service

# Start the service
systemctl start essentia-tagger.service

# Check status
systemctl status essentia-tagger.service
```

### Step 9: Verify It's Working

```bash
# Watch the logs in real-time
journalctl -u essentia-tagger.service -f

# In another terminal, or use Picard to save a file
# You should see the watcher detect and process it
```

## Configuration Reference

### Genre Format Styles

| Style | Example Output |
|-------|----------------|
| `parent_child` | "Rock - Alternative Rock" |
| `child_parent` | "Alternative Rock - Rock" |
| `child_only` | "Alternative Rock" |
| `raw` | "Rock---Alternative Rock" |

### Threshold Guidelines

**Genre Threshold (%):**
- `5-10%` - Very inclusive (more genres, lower confidence)
- `15%` - Balanced (recommended)
- `25-35%` - Strict (fewer, higher confidence genres)

**Mood Threshold (%):**
- `0.1-0.5%` - Inclusive (moods have naturally low confidence)
- `1%` - Balanced
- `3%+` - Strict (may get few/no moods)

### Command Line Arguments for tag_music.py

```
python tag_music.py [PATH] [OPTIONS]

Positional:
  PATH                    Path to music file or directory

Mode Options:
  --auto, -a              Run in automated (non-interactive) mode
  --single-file, -f       Process a single file (for file watcher)

Genre Options:
  --genres, -g N          Number of genres (default: 3)
  --genre-threshold, -gt  Confidence threshold % (default: 15)
  --genre-format, -gf     Format style (default: parent_child)

Mood Options:
  --no-moods              Disable mood analysis
  --mood-threshold, -mt   Confidence threshold % (default: 0.5)

Other Options:
  --dry-run, -d           Don't write tags
  --overwrite, -o         Overwrite existing tags
  --quiet, -q             Minimal output
  --log-dir DIR           Log file directory
  --model-dir DIR         Essentia models directory
```

## Troubleshooting

### Service Won't Start

```bash
# Check detailed status
systemctl status essentia-tagger.service -l

# Check full logs
journalctl -u essentia-tagger.service --no-pager

# Common fixes:
# 1. Check paths exist
ls -la /opt/essentia-tagger/
ls -la /opt/essentia-tagger/venv/
ls -la /opt/essentia-tagger/models/

# 2. Check permissions
chown -R root:root /opt/essentia-tagger
chmod +x /opt/essentia-tagger/essentia_watcher.sh

# 3. Test manually
/opt/essentia-tagger/essentia_watcher.sh --check
```

### Files Not Being Detected

```bash
# Check inotifywait is watching
ps aux | grep inotify

# Check watch directory path is correct
ls -la "/srv/dev-disk-by-uuid-dc4918d5-6597-465b-9567-ce442fbd8e2a/Media/Audio/Music/Sources/Clean"

# Increase inotify limits if needed
echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.conf
sysctl -p
```

### Essentia Errors

```bash
# Test essentia installation
source /opt/essentia-tagger/venv/bin/activate
python -c "from essentia.standard import MonoLoader; print('OK')"

# Test model loading
python -c "
from essentia.standard import TensorflowPredictEffnetDiscogs
model = TensorflowPredictEffnetDiscogs(
    graphFilename='/opt/essentia-tagger/models/discogs-effnet-bs64-1.pb',
    output='PartitionedCall:1'
)
print('Model loaded OK')
"
```

### High CPU Usage

The tagger uses TensorFlow which can be CPU-intensive. To reduce impact:

1. Increase debounce time:
   ```ini
   Environment="DEBOUNCE_SECONDS=10"
   ```

2. Process fewer genres:
   ```ini
   Environment="GENRES=2"
   ```

3. Apply CPU limits via systemd:
   ```ini
   [Service]
   CPUQuota=50%
   ```

## Managing the Service

```bash
# Start the service
systemctl start essentia-tagger

# Stop the service
systemctl stop essentia-tagger

# Restart after config changes
systemctl restart essentia-tagger

# Check status
systemctl status essentia-tagger

# View logs
journalctl -u essentia-tagger -f

# Disable auto-start
systemctl disable essentia-tagger
```

## View Logs

```bash
# Real-time service logs
journalctl -u essentia-tagger -f

# Tagger logs (detailed file-by-file)
ls -la /var/log/essentia-tagger/
tail -f /var/log/essentia-tagger/essentia_tagger_*.log
```

## File Structure

After setup, you should have:

```
/opt/essentia-tagger/
├── tag_music.py              # Main tagger script
├── essentia_watcher.sh       # File watcher script
├── venv/                     # Python virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
└── models/                   # Essentia ML models
    ├── discogs-effnet-bs64-1.pb
    ├── genre_discogs400-discogs-effnet-1.pb
    ├── genre_discogs400-discogs-effnet-1.json
    ├── mtg_jamendo_moodtheme-discogs-effnet-1.pb
    └── mtg_jamendo_moodtheme-discogs-effnet-1.json

/etc/systemd/system/
└── essentia-tagger.service   # Systemd service file

/var/log/essentia-tagger/
└── essentia_tagger_*.log     # Processing logs
```

## Workflow Summary

1. You load music in **Picard** (via web UI at port 5801)
2. You tag and save files in Picard
3. Picard moves files to `/storage/Media/Audio/Music/Sources/Clean`
4. The **file watcher** detects the new files
5. After a 5-second debounce, **tag_music.py** analyzes each file
6. Genre and mood tags are written to the file metadata
7. You can verify tags in Picard or any music player

## Quick Reference Card

```bash
# Check if service is running
systemctl status essentia-tagger

# View real-time logs
journalctl -u essentia-tagger -f

# Restart after config change
systemctl restart essentia-tagger

# Test a single file manually
source /opt/essentia-tagger/venv/bin/activate
python /opt/essentia-tagger/tag_music.py "/path/to/file.flac" --auto --single-file --model-dir /opt/essentia-tagger/models
deactivate

# Process a directory manually
source /opt/essentia-tagger/venv/bin/activate
python /opt/essentia-tagger/tag_music.py "/path/to/dir" --auto --model-dir /opt/essentia-tagger/models
deactivate
```
