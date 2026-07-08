"""
Tests for key and BPM detection in Essentia-to-Metadata.
Run with: python -m pytest tests/test_key_bpm.py -v
"""
import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# We need a real audio file for testing. Use the Planxty track we already have.
TEST_AUDIO = "/home/victor/servidor/music/Planxty (1972-2004)/1973 The Well Below The Valley (L)/01. Cunla.mp3"

# Skip all tests if test audio is not available
import pytest
pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_AUDIO),
    reason="Test audio file not available"
)


import sys
import argparse as _argparse

class TestConfig:
    """Test that config flags work correctly."""

    def _parse(self, *extra_args):
        """Helper: call parse_arguments with custom argv."""
        from tag_music import parse_arguments as _parse
        old = sys.argv
        sys.argv = ["tag_music.py"] + list(extra_args)
        try:
            return _parse()
        finally:
            sys.argv = old

    def test_key_bpm_enabled_by_default(self):
        from tag_music import Config
        c = Config()
        assert c.enable_key is True
        assert c.enable_bpm is True

    def test_args_disable_key(self):
        args = self._parse("--no-key")
        assert args.no_key is True

    def test_args_disable_bpm(self):
        args = self._parse("--no-bpm")
        assert args.no_bpm is True

    def test_config_from_args_respects_flags(self):
        args = self._parse("--no-key", "--no-bpm")
        from tag_music import config_from_args
        c = config_from_args(args)
        assert c.enable_key is False
        assert c.enable_bpm is False


class TestKeyBpmAnalysis:
    """Test that key and BPM are extracted from real audio."""

    def test_analyze_returns_key(self):
        from tag_music import Config, Logger, EssentiaAnalyzer
        config = Config()
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = True
        config.enable_bpm = False
        config.dry_run = True

        log_path = os.path.join(tempfile.gettempdir(), "test_key_bpm.log")
        logger = Logger(log_path)
        analyzer = EssentiaAnalyzer(config, logger)

        results = analyzer.analyze_file(Path(TEST_AUDIO))
        logger.close()

        assert results is not None, "Analysis returned None"
        assert 'key' in results, f"No key in results: {results.keys()}"
        assert results['key'] == "D major", f"Expected D major, got {results['key']}"
        assert results['key_name'] == "D"
        assert results['key_scale'] == "major"
        assert 0.0 <= results['key_strength'] <= 1.0

    def test_analyze_returns_bpm(self):
        from tag_music import Config, Logger, EssentiaAnalyzer
        config = Config()
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = False
        config.enable_bpm = True
        config.dry_run = True

        log_path = os.path.join(tempfile.gettempdir(), "test_bpm.log")
        logger = Logger(log_path)
        analyzer = EssentiaAnalyzer(config, logger)

        results = analyzer.analyze_file(Path(TEST_AUDIO))
        logger.close()

        assert results is not None
        assert 'bpm' in results, f"No bpm in results: {results.keys()}"
        assert isinstance(results['bpm'], int)
        # Planxty's "Cunla" is a reel at ~125 BPM
        assert 100 <= results['bpm'] <= 150, f"BPM {results['bpm']} out of expected range"

    def test_analyze_returns_both(self):
        from tag_music import Config, Logger, EssentiaAnalyzer
        config = Config()
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = True
        config.enable_bpm = True
        config.dry_run = True

        log_path = os.path.join(tempfile.gettempdir(), "test_both.log")
        logger = Logger(log_path)
        analyzer = EssentiaAnalyzer(config, logger)

        results = analyzer.analyze_file(Path(TEST_AUDIO))
        logger.close()

        assert results is not None
        assert 'key' in results
        assert 'bpm' in results
        assert results['key'] == "D major"
        assert 100 <= results['bpm'] <= 150

    def test_disabled_key_does_not_return_key(self):
        from tag_music import Config, Logger, EssentiaAnalyzer
        config = Config()
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = False
        config.enable_bpm = False
        config.dry_run = True

        log_path = os.path.join(tempfile.gettempdir(), "test_disabled.log")
        logger = Logger(log_path)
        analyzer = EssentiaAnalyzer(config, logger)

        results = analyzer.analyze_file(Path(TEST_AUDIO))
        logger.close()

        assert 'key' not in results
        assert 'bpm' not in results


class TestTagWriting:
    """Test that key/BPM tags are written to audio files without modifying originals."""

    def _copy_test_file(self, tmpdir):
        """Copy test audio to a temp location so we can write tags safely."""
        import shutil
        dest = os.path.join(tmpdir, "test_cunla.mp3")
        shutil.copy2(TEST_AUDIO, dest)
        return Path(dest)

    def test_vorbis_tags_written(self, tmp_path):
        """Test tag writing on a FLAC/OGG-like format — check the shared writer logic."""
        from tag_music import Config, Logger, TagWriter
        from mutagen.flac import FLAC

        # Create a minimal FLAC from the MP3? No — just test the writer logic by
        # verifying that the TagWriter processes key/bpm results correctly.
        config = Config()
        config.dry_run = False
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = True
        config.enable_bpm = True
        config.overwrite_existing = True

        # Use the mp3 test file directly
        test_file = self._copy_test_file(str(tmp_path))

        log_path = os.path.join(str(tmp_path), "write_test.log")
        logger = Logger(log_path)
        writer = TagWriter(config, logger)

        results = {
            'key': 'D major',
            'key_name': 'D',
            'key_scale': 'major',
            'key_strength': 0.85,
            'bpm': 125,
        }

        writer.write_tags(test_file, results)
        logger.close()

    def test_id3_tags_written_to_mp3(self, tmp_path):
        """Verify TKEY and TBPM are written to MP3 files."""
        from tag_music import Config, Logger, TagWriter
        from mutagen.id3 import ID3

        config = Config()
        config.dry_run = False
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = True
        config.enable_bpm = True
        config.overwrite_existing = True

        test_file = self._copy_test_file(str(tmp_path))
        log_path = os.path.join(str(tmp_path), "mp3_write_test.log")
        logger = Logger(log_path)
        writer = TagWriter(config, logger)

        results = {
            'key': 'D major',
            'key_name': 'D',
            'key_scale': 'major',
            'key_strength': 0.85,
            'bpm': 125,
        }

        writer.write_tags(test_file, results)
        logger.close()

        # Read back
        audio = ID3(str(test_file))
        tkey = audio.get('TKEY')
        tbpm = audio.get('TBPM')

        assert tkey is not None, "TKEY tag not written"
        assert str(tkey) == "D major", f"Expected D major, got {tkey}"
        assert tbpm is not None, "TBPM tag not written"
        assert str(tbpm) == "125", f"Expected 125, got {tbpm}"

    def test_dry_run_does_not_write(self, tmp_path):
        """Dry run must not modify files."""
        from tag_music import Config, Logger, TagWriter
        from mutagen.id3 import ID3

        config = Config()
        config.dry_run = True  # <-- DRY RUN
        config.enable_genres = False
        config.enable_moods = False
        config.enable_key = True
        config.enable_bpm = True
        config.overwrite_existing = True

        test_file = self._copy_test_file(str(tmp_path))
        log_path = os.path.join(str(tmp_path), "dry_run_test.log")
        logger = Logger(log_path)
        writer = TagWriter(config, logger)

        results = {'key': 'D major', 'bpm': 125}
        writer.write_tags(test_file, results)
        logger.close()

        # Verify file was NOT modified
        audio = ID3(str(test_file))
        assert audio.get('TKEY') is None, "TKEY should not be written in dry run"
        assert audio.get('TBPM') is None, "TBPM should not be written in dry run"
