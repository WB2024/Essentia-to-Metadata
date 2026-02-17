#!/bin/bash
# =============================================================================
# Essentia File Watcher for MusicBrainz Picard Integration
# =============================================================================
# Watches a directory for new/moved audio files and triggers essentia tagging.
# Designed to work with MusicBrainz Picard running in Docker on OMV7.
#
# Usage:
#   ./essentia_watcher.sh [options]
#
# Environment variables (or edit defaults below):
#   WATCH_DIR       - Directory to watch for new files
#   TAGGER_SCRIPT   - Path to tag_music.py
#   VENV_PATH       - Path to Python virtual environment
#   MODEL_DIR       - Path to Essentia models
#   LOG_DIR         - Directory for log files
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - Edit these to match your setup
# =============================================================================

# Directory to watch (where Picard moves files to)
WATCH_DIR="${WATCH_DIR:-/srv/dev-disk-by-uuid-dc4918d5-6597-465b-9567-ce442fbd8e2a/Media/Audio/Music/Sources/Clean}"

# Path to the tagger script
TAGGER_SCRIPT="${TAGGER_SCRIPT:-/opt/essentia-tagger/tag_music.py}"

# Python virtual environment path
VENV_PATH="${VENV_PATH:-/opt/essentia-tagger/venv}"

# Essentia models directory
MODEL_DIR="${MODEL_DIR:-/opt/essentia-tagger/models}"

# Log directory
LOG_DIR="${LOG_DIR:-/var/log/essentia-tagger}"

# Debounce time in seconds (wait for file operations to settle)
DEBOUNCE_SECONDS="${DEBOUNCE_SECONDS:-5}"

# Number of genres to tag
GENRES="${GENRES:-3}"

# Genre confidence threshold (percentage)
GENRE_THRESHOLD="${GENRE_THRESHOLD:-15}"

# Mood confidence threshold (percentage)
MOOD_THRESHOLD="${MOOD_THRESHOLD:-0.5}"

# Genre format: parent_child, child_parent, child_only, raw
GENRE_FORMAT="${GENRE_FORMAT:-parent_child}"

# Set to "true" for dry run mode (no tags written)
DRY_RUN="${DRY_RUN:-false}"

# Set to "true" to overwrite existing tags
OVERWRITE="${OVERWRITE:-true}"

# Audio file extensions to watch
AUDIO_EXTENSIONS="flac|mp3|ogg|m4a|wav"

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v inotifywait &> /dev/null; then
        log_error "inotifywait not found. Install with: apt install inotify-tools"
        exit 1
    fi
    
    if [ ! -f "$TAGGER_SCRIPT" ]; then
        log_error "Tagger script not found: $TAGGER_SCRIPT"
        exit 1
    fi
    
    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment not found: $VENV_PATH"
        log_info "Create it with: python3 -m venv $VENV_PATH"
        exit 1
    fi
    
    if [ ! -d "$MODEL_DIR" ]; then
        log_error "Model directory not found: $MODEL_DIR"
        exit 1
    fi
    
    if [ ! -d "$WATCH_DIR" ]; then
        log_error "Watch directory not found: $WATCH_DIR"
        exit 1
    fi
    
    # Create log directory if needed
    mkdir -p "$LOG_DIR"
    
    log "All dependencies OK"
}

# Build the tagger command arguments
build_tagger_args() {
    local filepath="$1"
    local args=""
    
    args="--auto"
    args="$args --single-file"
    args="$args --genres $GENRES"
    args="$args --genre-threshold $GENRE_THRESHOLD"
    args="$args --mood-threshold $MOOD_THRESHOLD"
    args="$args --genre-format $GENRE_FORMAT"
    args="$args --model-dir $MODEL_DIR"
    args="$args --log-dir $LOG_DIR"
    
    if [ "$DRY_RUN" = "true" ]; then
        args="$args --dry-run"
    fi
    
    if [ "$OVERWRITE" = "true" ]; then
        args="$args --overwrite"
    fi
    
    echo "$args"
}

# Process a single file
process_file() {
    local filepath="$1"
    
    # Check if file still exists (might have been moved again)
    if [ ! -f "$filepath" ]; then
        log_warn "File no longer exists: $filepath"
        return 0
    fi
    
    # Check file extension
    local ext="${filepath##*.}"
    ext="${ext,,}"  # lowercase
    
    if [[ ! "$ext" =~ ^($AUDIO_EXTENSIONS)$ ]]; then
        log_info "Skipping non-audio file: $filepath"
        return 0
    fi
    
    log "Processing: $filepath"
    
    # Build arguments
    local args=$(build_tagger_args "$filepath")
    
    # Activate virtual environment and run tagger
    (
        source "$VENV_PATH/bin/activate"
        python3 "$TAGGER_SCRIPT" "$filepath" $args
    )
    
    local status=$?
    if [ $status -eq 0 ]; then
        log "Successfully tagged: $(basename "$filepath")"
    else
        log_error "Failed to tag: $filepath (exit code: $status)"
    fi
    
    return $status
}

# Queue for debouncing
declare -A FILE_QUEUE
declare -A FILE_TIMESTAMPS

# Process the queue
process_queue() {
    local current_time=$(date +%s)
    
    for filepath in "${!FILE_QUEUE[@]}"; do
        local file_time="${FILE_TIMESTAMPS[$filepath]}"
        local age=$((current_time - file_time))
        
        if [ $age -ge $DEBOUNCE_SECONDS ]; then
            process_file "$filepath"
            unset FILE_QUEUE["$filepath"]
            unset FILE_TIMESTAMPS["$filepath"]
        fi
    done
}

# Add file to queue
queue_file() {
    local filepath="$1"
    FILE_QUEUE["$filepath"]=1
    FILE_TIMESTAMPS["$filepath"]=$(date +%s)
}

# Main watch loop
watch_directory() {
    log "Starting file watcher..."
    log "Watching: $WATCH_DIR"
    log "Tagger: $TAGGER_SCRIPT"
    log "Models: $MODEL_DIR"
    log "Logs: $LOG_DIR"
    log "Settings: genres=$GENRES, threshold=$GENRE_THRESHOLD%, format=$GENRE_FORMAT"
    log "Debounce: ${DEBOUNCE_SECONDS}s"
    echo ""
    log "Waiting for new files..."
    
    # Use inotifywait to watch for file events
    # -m = monitor mode (continuous)
    # -r = recursive
    # -e = events to watch
    # --format = output format
    inotifywait -m -r -e moved_to -e close_write --format '%w%f' "$WATCH_DIR" 2>/dev/null | while read filepath; do
        # Check if it's an audio file by extension
        local ext="${filepath##*.}"
        ext="${ext,,}"
        
        if [[ "$ext" =~ ^($AUDIO_EXTENSIONS)$ ]]; then
            log_info "Detected: $filepath"
            
            # Wait for debounce period to let file operations settle
            sleep "$DEBOUNCE_SECONDS"
            
            # Process the file
            process_file "$filepath"
        fi
    done
}

# Print help
show_help() {
    cat << EOF
Essentia File Watcher for MusicBrainz Picard Integration

Usage: $0 [options]

Options:
    -h, --help      Show this help message
    -c, --check     Check dependencies only
    -t, --test      Test mode - process existing files then exit
    -d, --dry-run   Enable dry run mode (no tags written)

Environment Variables:
    WATCH_DIR       Directory to watch (default: $WATCH_DIR)
    TAGGER_SCRIPT   Path to tag_music.py
    VENV_PATH       Path to Python venv
    MODEL_DIR       Path to Essentia models
    LOG_DIR         Directory for logs
    DEBOUNCE_SECONDS  Wait time before processing (default: 5)
    GENRES          Number of genres (default: 3)
    GENRE_THRESHOLD Genre confidence % (default: 15)
    MOOD_THRESHOLD  Mood confidence % (default: 0.5)
    GENRE_FORMAT    Format style (default: parent_child)
    DRY_RUN         Set to 'true' for dry run
    OVERWRITE       Set to 'true' to overwrite existing tags

Examples:
    # Start watching
    $0
    
    # Check dependencies
    $0 --check
    
    # Test with dry run
    DRY_RUN=true $0 --test
    
    # Override settings
    GENRES=4 GENRE_THRESHOLD=20 $0

EOF
}

# Test mode - process existing files
test_mode() {
    log "Test mode - scanning for existing audio files..."
    
    find "$WATCH_DIR" -type f \( -iname "*.flac" -o -iname "*.mp3" -o -iname "*.ogg" -o -iname "*.m4a" -o -iname "*.wav" \) | head -5 | while read filepath; do
        process_file "$filepath"
    done
    
    log "Test complete"
}

# Main entry point
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check)
            check_dependencies
            exit 0
            ;;
        -t|--test)
            check_dependencies
            test_mode
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            check_dependencies
            watch_directory
            ;;
        "")
            check_dependencies
            watch_directory
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Trap signals for clean shutdown
trap 'log "Shutting down..."; exit 0' SIGTERM SIGINT

# Run main
main "$@"
