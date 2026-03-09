#!/usr/bin/env python3
"""
Analyze music library with Essentia and write genre/mood to tags
Supports both interactive mode and CLI arguments for automation
"""
import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import platform
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TCON, COMM
from mutagen.oggvorbis import OggVorbis
from mutagen.oggopus import OggOpus
from mutagen.mp4 import MP4
from mutagen.aiff import AIFF
from mutagen.wavpack import WavPack
from mutagen.musepack import Musepack
from mutagen.apev2 import APEv2
from mutagen.asf import ASF

# Supported audio file extensions
# Analysis: all formats Essentia MonoLoader can decode (via FFmpeg)
# Tag writing: each format uses an appropriate tag writer
AUDIO_EXTENSIONS = {
    '.flac',                    # FLAC - Vorbis comments
    '.mp3',                     # MP3 - ID3v2
    '.ogg', '.oga',             # Ogg Vorbis - Vorbis comments
    '.opus',                    # Opus - Vorbis comments
    '.m4a', '.m4b', '.mp4', '.aac',  # AAC/ALAC - MP4 atoms
    '.wma',                     # WMA - ASF attributes
    '.aiff', '.aif',            # AIFF - ID3v2
    '.wav',                     # WAV - ID3v2 (via mutagen)
    '.wv',                      # WavPack - APEv2
    '.ape',                     # Monkey's Audio - APEv2
    '.mpc', '.mp+',             # Musepack - APEv2
    '.dsf',                     # DSD Stream File - ID3v2
}

# Model directory (fixed)
MODEL_DIR = os.path.expanduser('~/essentia_models')

# Model files
EMBEDDING_MODEL = f"{MODEL_DIR}/discogs-effnet-bs64-1.pb"
GENRE_MODEL = f"{MODEL_DIR}/genre_discogs400-discogs-effnet-1.pb"
GENRE_METADATA = f"{MODEL_DIR}/genre_discogs400-discogs-effnet-1.json"
MOOD_MODEL = f"{MODEL_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
MOOD_METADATA = f"{MODEL_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.json"


def format_genre_tag(raw_genre, style='parent_child'):
    """
    Format genre tags for better readability
    
    Args:
        raw_genre: Raw genre string like "Rock---Alternative Rock"
        style: Formatting style
            - 'parent_child': "Rock - Alternative Rock" (default)
            - 'child_parent': "Alternative Rock - Rock"
            - 'child_only': "Alternative Rock"
            - 'raw': "Rock---Alternative Rock" (no formatting)
    
    Returns:
        Formatted genre string
    """
    if style == 'raw':
        return raw_genre
    
    # Split on triple dash
    if '---' in raw_genre:
        parts = raw_genre.split('---')
        parent = parts[0].strip()
        child = parts[1].strip() if len(parts) > 1 else ''
        
        if style == 'parent_child':
            # "Rock - Alternative Rock"
            if child:
                return f"{parent} - {child}"
            else:
                return parent
        
        elif style == 'child_parent':
            # "Alternative Rock - Rock"
            if child:
                return f"{child} - {parent}"
            else:
                return parent
        
        elif style == 'child_only':
            # "Alternative Rock" (just the specific genre)
            return child if child else parent
    
    # No triple dash, return as-is
    return raw_genre


def format_mood_tag(raw_mood):
    """
    Format mood tags for better readability
    
    Args:
        raw_mood: Raw mood string
    
    Returns:
        Formatted mood string (capitalized, etc.)
    """
    # Capitalize first letter of each word
    return raw_mood.title()


class Logger:
    """Dual output to console and log file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handle = open(log_file, 'w', encoding='utf-8')
        self.write_header()
    
    def write_header(self):
        """Write log file header"""
        header = f"""
{'=' * 80}
ESSENTIA MUSIC TAGGER - LOG FILE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

"""
        self.file_handle.write(header)
        self.file_handle.flush()
    
    def log(self, message, console=True, file=True):
        """Write to console and/or file"""
        if console:
            print(message)
        if file:
            self.file_handle.write(message + '\n')
            self.file_handle.flush()
    
    def log_config(self, config, music_path):
        """Log configuration"""
        if config.enable_genres and config.enable_moods:
            mode_label = "Genres & Moods"
        elif config.enable_genres:
            mode_label = "Genres only"
        else:
            mode_label = "Moods only"

        config_text = f"""
CONFIGURATION:
{'-' * 80}
Target Directory: {music_path}
Model Directory: {MODEL_DIR}
Analysis Mode: {mode_label}
"""
        if config.enable_genres:
            config_text += f"""Genre Settings:
  - Number of genres: {config.top_n_genres}
  - Confidence threshold: {config.genre_threshold:.1%}
  - Genre format: {config.genre_format}
"""
        if config.enable_moods:
            config_text += f"""Mood Settings:
  - Confidence threshold: {config.mood_threshold:.1%}
"""
        config_text += f"""Other Settings:
  - Dry run mode: {config.dry_run}
  - Write confidence tags: {config.write_confidence_tags}
  - Overwrite existing: {config.overwrite_existing}
  - Verbose output: {config.verbose}
{'=' * 80}

"""
        self.file_handle.write(config_text)
        self.file_handle.flush()
    
    def log_analysis(self, filepath, results, relative_path):
        """Log detailed analysis results"""
        log_entry = f"""
FILE: {relative_path}
{'-' * 80}
"""
        
        # Genre results (raw)
        if results.get('genres'):
            log_entry += "GENRES (raw predictions):\n"
            for g in results['genres']:
                log_entry += f"  • {g['label']}: {g['confidence']:.2%}\n"
        elif 'genres' not in results:
            log_entry += "GENRES: Disabled\n"
        else:
            log_entry += "GENRES: None passed threshold\n"
        
        # Formatted genres
        if results.get('formatted_genres'):
            log_entry += "\nGENRES (formatted for tags):\n"
            for fg in results['formatted_genres']:
                log_entry += f"  • {fg}\n"
        
        # All genre predictions (top 10)
        if results.get('all_genres_debug'):
            log_entry += "\nALL GENRE PREDICTIONS (top 10):\n"
            for label, conf in results['all_genres_debug']:
                log_entry += f"  • {label}: {conf:.2%}\n"
        
        # Mood results (raw)
        if results.get('moods'):
            log_entry += f"\nMOODS (raw predictions - {len(results['moods'])} total):\n"
            for m in results['moods']:
                log_entry += f"  • {m['label']}: {m['confidence']:.2%}\n"
        elif 'moods' not in results:
            log_entry += "\nMOODS: Disabled\n"
        else:
            log_entry += "\nMOODS: None passed threshold\n"
        
        # Formatted moods
        if results.get('formatted_moods'):
            log_entry += "\nMOODS (formatted for tags):\n"
            for fm in results['formatted_moods']:
                log_entry += f"  • {fm}\n"
        
        # All mood predictions (top 10)
        if results.get('all_moods_debug'):
            log_entry += "\nALL MOOD PREDICTIONS (top 10):\n"
            for label, conf in results['all_moods_debug'][:10]:
                log_entry += f"  • {label}: {conf:.2%}\n"
        
        log_entry += f"\n{'=' * 80}\n"
        
        self.file_handle.write(log_entry)
        self.file_handle.flush()
    
    def log_summary(self, processed, errors, skipped):
        """Log final summary"""
        summary = f"""
{'=' * 80}
PROCESSING SUMMARY
{'=' * 80}
Total Processed: {processed}
Errors: {errors}
Skipped: {skipped}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
        self.file_handle.write(summary)
        self.file_handle.flush()
    
    def close(self):
        """Close log file"""
        self.file_handle.close()


# Persistent settings file for storing default library path
SETTINGS_FILE = os.path.expanduser('~/.essentia_tagger.json')


def load_settings():
    """Load persistent settings (default library path, etc.)"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_settings(settings):
    """Save persistent settings to disk"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except IOError as e:
        print(f"   \u26a0\ufe0f  Could not save settings: {e}")


class Config:
    """Runtime configuration from user prompts"""
    def __init__(self):
        self.dry_run = True
        self.enable_genres = True
        self.enable_moods = True
        self.top_n_genres = 3
        self.genre_threshold = 0.15
        self.mood_threshold = 0.005
        self.write_confidence_tags = True
        self.overwrite_existing = False
        self.verbose = True
        self.log_file = None
        self.genre_format = 'parent_child'
        self.default_library_path = None


class EssentiaAnalyzer:
    """Analyze audio files with Essentia models"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        logger.log("\n🔄 Loading models...")
        
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=EMBEDDING_MODEL,
            output="PartitionedCall:1"
        )
        
        # Conditionally load genre model
        self.genre_model = None
        self.genre_labels = None
        if config.enable_genres:
            self.genre_model = TensorflowPredict2D(
                graphFilename=GENRE_MODEL,
                input="serving_default_model_Placeholder",
                output="PartitionedCall"
            )
            with open(GENRE_METADATA, 'r') as f:
                metadata = json.load(f)
                self.genre_labels = metadata['classes']
            logger.log(f"   ✅ Loaded {len(self.genre_labels)} genre classes")
        else:
            logger.log("   ⏭️  Genre analysis disabled by user")
        
        # Conditionally load mood model
        self.mood_model = None
        self.mood_labels = None
        if config.enable_moods and os.path.exists(MOOD_MODEL):
            logger.log("   Loading mood model...")
            try:
                self.mood_model = TensorflowPredict2D(
                    graphFilename=MOOD_MODEL,
                    input="model/Placeholder",
                    output="model/Sigmoid"
                )
                with open(MOOD_METADATA, 'r') as f:
                    metadata = json.load(f)
                    self.mood_labels = metadata['classes']
                logger.log(f"   ✅ Loaded {len(self.mood_labels)} mood classes")
            except Exception as e:
                logger.log(f"   ⚠️  Could not load mood model: {e}")
                logger.log("      Continuing without mood analysis...")
                self.mood_model = None
        elif not config.enable_moods:
            logger.log("   ⏭️  Mood analysis disabled by user")
        else:
            logger.log("   ⚠️  Mood model not found")
        
        logger.log("   ✅ Models loaded successfully!\n")
    
    def analyze_file(self, filepath):
        """Analyze a single audio file"""
        try:
            # Load audio (resampled to 16kHz)
            audio = MonoLoader(
                filename=str(filepath),
                sampleRate=16000,
                resampleQuality=4
            )()
            
            # Get embeddings
            embeddings = self.embedding_model(audio)
            
            results = {}
            
            # Predict genres (if model loaded)
            if self.genre_model:
                genre_predictions = self.genre_model(embeddings)
                genre_activations = np.mean(genre_predictions, axis=0)
                
                # Get top genres with threshold
                top_indices = np.argsort(genre_activations)[::-1][:self.config.top_n_genres * 2]
                genres = []
                for idx in top_indices:
                    if len(genres) >= self.config.top_n_genres:
                        break
                    if genre_activations[idx] >= self.config.genre_threshold:
                        genres.append({
                            'label': self.genre_labels[idx],
                            'confidence': float(genre_activations[idx])
                        })
                
                # If no genres pass threshold, take top 1 anyway
                if not genres:
                    top_idx = np.argmax(genre_activations)
                    genres.append({
                        'label': self.genre_labels[top_idx],
                        'confidence': float(genre_activations[top_idx])
                    })
                
                results['genres'] = genres
                
                # Format genres for tag writing
                results['formatted_genres'] = [
                    format_genre_tag(g['label'], style=self.config.genre_format) 
                    for g in genres
                ]
                
                # Store all genre activations for logging
                all_top_indices = np.argsort(genre_activations)[::-1][:10]
                results['all_genres_debug'] = [
                    (self.genre_labels[idx], float(genre_activations[idx])) 
                    for idx in all_top_indices
                ]
            
            # Predict moods (if model loaded)
            if self.mood_model:
                mood_predictions = self.mood_model(embeddings)
                mood_activations = np.mean(mood_predictions, axis=0)
                
                # Log raw mood activation stats
                max_mood = np.max(mood_activations)
                mean_mood = np.mean(mood_activations)
                
                self.logger.log(f"     [MOOD DEBUG] Max: {max_mood:.4f}, Mean: {mean_mood:.4f}, Threshold: {self.config.mood_threshold:.4f}", console=False)
                
                moods = []
                for idx, activation in enumerate(mood_activations):
                    if activation >= self.config.mood_threshold:
                        moods.append({
                            'label': self.mood_labels[idx],
                            'confidence': float(activation)
                        })
                
                # Sort by confidence
                moods = sorted(moods, key=lambda x: x['confidence'], reverse=True)
                results['moods'] = moods[:5]  # Limit to top 5 moods
                
                # Format moods for tag writing
                results['formatted_moods'] = [
                    format_mood_tag(m['label']) 
                    for m in results['moods']
                ]
                
                # Store all mood activations for logging
                results['all_moods_debug'] = sorted(
                    [(self.mood_labels[idx], float(mood_activations[idx])) 
                     for idx in range(len(mood_activations))],
                    key=lambda x: x[1], reverse=True
                )
                
                self.logger.log(f"     [MOOD DEBUG] Found {len(moods)} moods above threshold", console=False)
            
            return results
            
        except Exception as e:
            self.logger.log(f"     ⚠️  Error analyzing: {e}")
            return None


class TagWriter:
    """Write analysis results to audio file tags"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def write_tags(self, filepath, results):
        """Write genre/mood to file tags"""
        if self.config.dry_run:
            tag_info = []
            if self.config.enable_genres and results.get('formatted_genres'):
                tag_info.append(f"Genres: {', '.join(results['formatted_genres'])}")
            if self.config.enable_moods and results.get('formatted_moods'):
                tag_info.append(f"Moods: {', '.join(results['formatted_moods'][:3])}")
            self.logger.log(f"     [DRY RUN] Would write: {' | '.join(tag_info)}")
            return
        
        try:
            file_ext = filepath.suffix.lower()
            
            if file_ext == '.flac':
                self._write_flac(filepath, results)
            elif file_ext == '.mp3':
                self._write_mp3(filepath, results)
            elif file_ext in ('.ogg', '.oga'):
                self._write_ogg(filepath, results)
            elif file_ext == '.opus':
                self._write_opus(filepath, results)
            elif file_ext in ('.m4a', '.m4b', '.mp4', '.aac'):
                self._write_mp4(filepath, results)
            elif file_ext == '.wma':
                self._write_wma(filepath, results)
            elif file_ext in ('.aiff', '.aif'):
                self._write_aiff(filepath, results)
            elif file_ext in ('.wav', '.dsf'):
                self._write_id3_generic(filepath, results)
            elif file_ext in ('.wv', '.ape', '.mpc', '.mp+'):
                self._write_apev2(filepath, results)
            else:
                self.logger.log(f"     ⚠️  Unsupported format: {file_ext}")
                
        except Exception as e:
            self.logger.log(f"     ⚠️  Error writing tags: {e}")
    
    def _write_flac(self, filepath, results):
        """Write to FLAC tags (Vorbis comments)"""
        audio = FLAC(filepath)
        tags_written = self._write_vorbis_comments(audio, results)
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_mp3(self, filepath, results):
        """Write to MP3 ID3 tags"""
        try:
            audio = ID3(filepath)
        except Exception:
            audio = ID3()
        self._write_id3_tags(audio, results)
        audio.save(filepath)
    
    def _write_vorbis_comments(self, audio, results):
        """Shared writer for Vorbis-comment-based formats (FLAC, OGG, Opus)"""
        tags_written = []
        
        if self.config.enable_genres and results.get('formatted_genres'):
            if self.config.overwrite_existing or 'GENRE' not in audio:
                genre_str = '; '.join(results['formatted_genres'])
                audio['GENRE'] = genre_str
                tags_written.append(f"GENRE={genre_str}")
                
                if self.config.write_confidence_tags and results.get('genres'):
                    genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                    confidence_str = ', '.join(genre_details)
                    audio['ESSENTIA_GENRE'] = f"Essentia: {confidence_str}"
                    tags_written.append(f"ESSENTIA_GENRE={confidence_str}")
            else:
                self.logger.log("     ⏭️  Skipping genres (already has GENRE tag)")
        
        if self.config.enable_moods and results.get('formatted_moods'):
            if self.config.overwrite_existing or 'MOOD' not in audio:
                mood_str = '; '.join(results['formatted_moods'][:3])
                audio['MOOD'] = mood_str
                tags_written.append(f"MOOD={mood_str}")
                
                if self.config.write_confidence_tags and results.get('moods'):
                    mood_details = [f"{m['label']}: {m['confidence']:.2%}" for m in results['moods'][:3]]
                    mood_conf_str = ', '.join(mood_details)
                    audio['ESSENTIA_MOOD'] = f"Essentia: {mood_conf_str}"
                    tags_written.append(f"ESSENTIA_MOOD={mood_conf_str}")
            else:
                self.logger.log("     ⏭️  Skipping moods (already has MOOD tag)")
        
        return tags_written
    
    def _write_ogg(self, filepath, results):
        """Write to OGG Vorbis tags"""
        audio = OggVorbis(filepath)
        tags_written = self._write_vorbis_comments(audio, results)
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_opus(self, filepath, results):
        """Write to Opus tags"""
        audio = OggOpus(filepath)
        tags_written = self._write_vorbis_comments(audio, results)
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_mp4(self, filepath, results):
        """Write to MP4/M4A/AAC tags (iTunes-style atoms)"""
        audio = MP4(filepath)
        tags_written = []
        
        if self.config.enable_genres and results.get('formatted_genres'):
            has_existing = '\xa9gen' in audio.tags if audio.tags else False
            if self.config.overwrite_existing or not has_existing:
                genre_str = '; '.join(results['formatted_genres'])
                audio['\xa9gen'] = [genre_str]
                tags_written.append(f"genre={genre_str}")
                
                if self.config.write_confidence_tags and results.get('genres'):
                    genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                    confidence_str = ', '.join(genre_details)
                    audio['\xa9cmt'] = [f"Essentia Genre: {confidence_str}"]
                    tags_written.append(f"comment(genre)={confidence_str}")
            else:
                self.logger.log("     ⏭️  Skipping genres (already has genre tag)")
        
        if self.config.enable_moods and results.get('formatted_moods'):
            mood_str = '; '.join(results['formatted_moods'][:3])
            audio['----:com.apple.iTunes:MOOD'] = [
                mutagen.mp4.MP4FreeForm(mood_str.encode('utf-8'), dataformat=mutagen.mp4.AtomDataType.UTF8)
            ]
            tags_written.append(f"MOOD={mood_str}")
            
            if self.config.write_confidence_tags and results.get('moods'):
                mood_details = [f"{m['label']}: {m['confidence']:.2%}" for m in results['moods'][:3]]
                mood_conf_str = ', '.join(mood_details)
                audio['----:com.apple.iTunes:ESSENTIA_MOOD'] = [
                    mutagen.mp4.MP4FreeForm(mood_conf_str.encode('utf-8'), dataformat=mutagen.mp4.AtomDataType.UTF8)
                ]
                tags_written.append(f"ESSENTIA_MOOD={mood_conf_str}")
        
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_wma(self, filepath, results):
        """Write to WMA/ASF tags"""
        audio = ASF(filepath)
        tags_written = []
        
        if self.config.enable_genres and results.get('formatted_genres'):
            has_existing = 'WM/Genre' in audio if audio.tags else False
            if self.config.overwrite_existing or not has_existing:
                genre_str = '; '.join(results['formatted_genres'])
                audio['WM/Genre'] = genre_str
                tags_written.append(f"WM/Genre={genre_str}")
                
                if self.config.write_confidence_tags and results.get('genres'):
                    genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                    confidence_str = ', '.join(genre_details)
                    audio['ESSENTIA_GENRE'] = f"Essentia: {confidence_str}"
                    tags_written.append(f"ESSENTIA_GENRE={confidence_str}")
            else:
                self.logger.log("     ⏭️  Skipping genres (already has genre tag)")
        
        if self.config.enable_moods and results.get('formatted_moods'):
            mood_str = '; '.join(results['formatted_moods'][:3])
            audio['WM/Mood'] = mood_str
            tags_written.append(f"WM/Mood={mood_str}")
            
            if self.config.write_confidence_tags and results.get('moods'):
                mood_details = [f"{m['label']}: {m['confidence']:.2%}" for m in results['moods'][:3]]
                mood_conf_str = ', '.join(mood_details)
                audio['ESSENTIA_MOOD'] = f"Essentia: {mood_conf_str}"
                tags_written.append(f"ESSENTIA_MOOD={mood_conf_str}")
        
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_aiff(self, filepath, results):
        """Write to AIFF tags (ID3v2)"""
        audio = AIFF(filepath)
        if audio.tags is None:
            audio.add_tags()
        self._write_id3_tags(audio.tags, results)
        audio.save()
    
    def _write_id3_generic(self, filepath, results):
        """Write ID3v2 tags to WAV, DSF, etc. via mutagen.File"""
        audio = mutagen.File(filepath)
        if audio is None:
            self.logger.log("     ⚠️  Could not open file for tagging")
            return
        if audio.tags is None:
            audio.add_tags()
        self._write_id3_tags(audio.tags, results)
        audio.save()
    
    def _write_id3_tags(self, tags, results):
        """Shared ID3v2 tag writer used by MP3, AIFF, WAV, DSF"""
        tags_written = []
        
        if self.config.enable_genres and results.get('formatted_genres'):
            has_existing_genre = bool(tags.getall('TCON'))
            if self.config.overwrite_existing or not has_existing_genre:
                genre_str = '; '.join(results['formatted_genres'])
                tags.delall('TCON')
                tags.add(TCON(encoding=3, text=genre_str))
                tags_written.append(f"TCON={genre_str}")
                
                if self.config.write_confidence_tags and results.get('genres'):
                    genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                    confidence_str = ', '.join(genre_details)
                    tags.delall('COMM::eng')
                    tags.add(COMM(
                        encoding=3,
                        lang='eng',
                        desc='Essentia Genre',
                        text=confidence_str
                    ))
                    tags_written.append(f"COMM(genre)={confidence_str}")
            else:
                self.logger.log("     ⏭️  Skipping genres (already has GENRE tag)")
        
        if self.config.enable_moods and results.get('formatted_moods'):
            mood_str = '; '.join(results['formatted_moods'][:3])
            tags.add(COMM(
                encoding=3,
                lang='eng',
                desc='Essentia Mood',
                text=mood_str
            ))
            tags_written.append(f"COMM(mood)={mood_str}")
        
        if tags_written:
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_apev2(self, filepath, results):
        """Write APEv2 tags (WavPack, Monkey's Audio, Musepack)"""
        try:
            audio = mutagen.File(filepath)
            if audio is None:
                self.logger.log("     ⚠️  Could not open file for tagging")
                return
            if audio.tags is None:
                audio.add_tags()
        except Exception:
            self.logger.log("     ⚠️  Could not read/create APEv2 tags")
            return
        
        tags_written = []
        
        if self.config.enable_genres and results.get('formatted_genres'):
            has_existing = 'Genre' in audio.tags
            if self.config.overwrite_existing or not has_existing:
                genre_str = '; '.join(results['formatted_genres'])
                audio.tags['Genre'] = genre_str
                tags_written.append(f"Genre={genre_str}")
                
                if self.config.write_confidence_tags and results.get('genres'):
                    genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                    confidence_str = ', '.join(genre_details)
                    audio.tags['Essentia Genre'] = f"Essentia: {confidence_str}"
                    tags_written.append(f"Essentia Genre={confidence_str}")
            else:
                self.logger.log("     ⏭️  Skipping genres (already has Genre tag)")
        
        if self.config.enable_moods and results.get('formatted_moods'):
            mood_str = '; '.join(results['formatted_moods'][:3])
            audio.tags['Mood'] = mood_str
            tags_written.append(f"Mood={mood_str}")
            
            if self.config.write_confidence_tags and results.get('moods'):
                mood_details = [f"{m['label']}: {m['confidence']:.2%}" for m in results['moods'][:3]]
                mood_conf_str = ', '.join(mood_details)
                audio.tags['Essentia Mood'] = f"Essentia: {mood_conf_str}"
                tags_written.append(f"Essentia Mood={mood_conf_str}")
        
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)


def scan_library(root_path, analyzer, tag_writer, config, logger):
    """Recursively scan and process music library"""
    root = Path(root_path)
    
    logger.log("\n🔍 Scanning for audio files...")
    files = list(root.rglob('*'))
    audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    
    if not audio_files:
        logger.log("❌ No audio files found in this directory!")
        return
    
    logger.log(f"🎵 Found {len(audio_files)} audio files")
    logger.log(f"{'=' * 70}\n")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for i, filepath in enumerate(audio_files, 1):
        try:
            relative_path = filepath.relative_to(root)
        except ValueError:
            relative_path = filepath.name
        
        logger.log(f"[{i}/{len(audio_files)}] {relative_path}")
        
        # Analyze
        results = analyzer.analyze_file(filepath)
        
        if results:
            # Print genre results to console
            if config.enable_genres:
                if results.get('genres'):
                    genre_list = [f"{g['label']} ({g['confidence']:.1%})" for g in results['genres']]
                    logger.log(f"     🎸 Raw: {', '.join(genre_list)}")
                if results.get('formatted_genres'):
                    logger.log(f"     🎸 Formatted: {', '.join(results['formatted_genres'])}")
            
            # Print mood results to console
            if config.enable_moods:
                if results.get('moods'):
                    mood_list = [f"{m['label']} ({m['confidence']:.1%})" for m in results['moods'][:3]]
                    logger.log(f"     😊 Raw: {', '.join(mood_list)}")
                if results.get('formatted_moods'):
                    logger.log(f"     😊 Formatted: {', '.join(results['formatted_moods'][:3])}")
                elif results.get('moods') is not None and len(results.get('moods', [])) == 0:
                    logger.log(f"     😊 Moods: None above threshold ({config.mood_threshold:.2%})")
            
            # Show debug info if verbose
            if config.verbose and results.get('all_genres_debug'):
                top_5 = ', '.join([f"{label} ({conf:.1%})" for label, conf in results['all_genres_debug'][:5]])
                logger.log(f"     📊 Top 5 genres: {top_5}", console=False)
            
            # Log detailed analysis to file
            logger.log_analysis(filepath, results, relative_path)
            
            # Write tags
            tag_writer.write_tags(filepath, results)
            
            if not config.dry_run:
                logger.log("     ✅ Tags written")
            
            processed += 1
        else:
            errors += 1
        
        logger.log("")
    
    # Summary
    logger.log(f"\n{'=' * 70}")
    logger.log(f"📊 SUMMARY")
    logger.log(f"{'=' * 70}")
    logger.log(f"✅ Processed: {processed}")
    logger.log(f"❌ Errors: {errors}")
    logger.log(f"⏭️  Skipped: {skipped}")
    
    logger.log_summary(processed, errors, skipped)


def _read_key():
    """Read a single keypress, returning special keys as names.
    Returns: str - single char or 'up', 'down', 'enter', 'backspace', 'q'
    """
    if platform.system() == 'Windows':
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ('\r', '\n'):
            return 'enter'
        if ch == '\x08' or ch == '\x7f':
            return 'backspace'
        if ch in ('\x00', '\xe0'):  # special key prefix on Windows
            ch2 = msvcrt.getwch()
            if ch2 == 'H':
                return 'up'
            if ch2 == 'P':
                return 'down'
            return None
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\r' or ch == '\n':
                return 'enter'
            if ch == '\x7f' or ch == '\x08':
                return 'backspace'
            if ch == '\x1b':  # escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'up'
                    if ch3 == 'B':
                        return 'down'
                return None
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clear_lines(n):
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write('\x1b[A')   # move up
        sys.stdout.write('\x1b[2K')  # clear line
    sys.stdout.flush()


def browse_directory(start_path):
    """Interactive directory browser with arrow-key navigation.
    
    Args:
        start_path: Root directory to start browsing from
    
    Returns:
        str - selected directory path, or None if cancelled
    """
    current_path = Path(start_path)
    selected_idx = 0
    page_size = 15
    scroll_offset = 0
    
    while True:
        # Get subdirectories of current path
        try:
            subdirs = sorted(
                [d for d in current_path.iterdir() if d.is_dir()],
                key=lambda d: d.name.lower()
            )
        except PermissionError:
            print("\n   ⚠️  Permission denied. Going back...")
            current_path = current_path.parent
            selected_idx = 0
            scroll_offset = 0
            continue
        
        # Build menu items
        items = []
        items.append(('✅ SELECT THIS FOLDER', 'select'))
        if current_path != Path(start_path):
            items.append(('⬆️  ../ (go up)', 'up'))
        for d in subdirs:
            items.append((f"📁 {d.name}", str(d)))
        
        # Clamp selection
        if selected_idx >= len(items):
            selected_idx = len(items) - 1
        if selected_idx < 0:
            selected_idx = 0
        
        # Adjust scroll so selected item is visible
        if selected_idx < scroll_offset:
            scroll_offset = selected_idx
        if selected_idx >= scroll_offset + page_size:
            scroll_offset = selected_idx - page_size + 1
        
        visible_items = items[scroll_offset:scroll_offset + page_size]
        
        # Render
        lines = []
        rel_path = str(current_path)
        try:
            rel_path = str(current_path.relative_to(start_path))
            if rel_path == '.':
                rel_path = '(library root)'
            else:
                rel_path = f"/{rel_path}"
        except ValueError:
            pass
        
        lines.append(f"\n   📂 Browsing: {rel_path}")
        lines.append(f"   📍 Full path: {current_path}")
        lines.append("   Use ↑↓ arrows to navigate, Enter to select, 'q' to cancel")
        lines.append("   " + "─" * 50)
        
        for i, (label, _action) in enumerate(visible_items):
            global_idx = i + scroll_offset
            if global_idx == selected_idx:
                lines.append(f"   ▶ {label}")
            else:
                lines.append(f"     {label}")
        
        if scroll_offset > 0:
            lines.append(f"   ↑ ({scroll_offset} more above)")
        remaining_below = len(items) - scroll_offset - page_size
        if remaining_below > 0:
            lines.append(f"   ↓ ({remaining_below} more below)")
        
        lines.append("")
        
        output = '\n'.join(lines)
        sys.stdout.write(output)
        sys.stdout.flush()
        
        # Read key
        key = _read_key()
        
        # Clear the rendered block before re-rendering
        line_count = len(lines)
        _clear_lines(line_count)
        
        if key == 'up':
            if selected_idx > 0:
                selected_idx -= 1
        elif key == 'down':
            if selected_idx < len(items) - 1:
                selected_idx += 1
        elif key == 'enter':
            _label, action = items[selected_idx]
            if action == 'select':
                return str(current_path)
            elif action == 'up':
                current_path = current_path.parent
                selected_idx = 0
                scroll_offset = 0
            else:
                # Navigate into subfolder
                current_path = Path(action)
                selected_idx = 0
                scroll_offset = 0
        elif key == 'q':
            return None
        elif key == 'backspace':
            if current_path != Path(start_path):
                current_path = current_path.parent
                selected_idx = 0
                scroll_offset = 0


def get_music_path(config):
    """Prompt user for music directory path, with optional library browsing"""
    print("\n" + "=" * 70)
    print("🎸 ESSENTIA MUSIC TAGGER - INTERACTIVE MODE")
    print("=" * 70)
    print("\nThis tool will recursively analyze ALL audio files")
    print("in the directory you specify and its subdirectories.\n")
    
    library_path = config.default_library_path
    
    if library_path and os.path.isdir(library_path):
        print(f"📚 Default library: {library_path}")
        print()
        print("How would you like to choose the scan path?")
        print("   1 = Scan entire library (default)")
        print("   2 = Browse & select a folder within library")
        print("   3 = Enter a custom path")
        print()
        
        while True:
            choice = input("Select option [1]: ").strip()
            if choice in ('', '1'):
                # Scan entire library
                path = Path(library_path)
                sample_files = list(path.rglob('*'))
                audio_count = len([f for f in sample_files if f.suffix.lower() in AUDIO_EXTENSIONS])
                print(f"\n📂 Directory: {path}")
                print(f"🎵 Found ~{audio_count} audio files")
                confirm = input("\nProceed with this directory? [Y/n]: ").strip().lower()
                if confirm in ('', 'y', 'yes'):
                    return str(path)
                else:
                    print("Cancelled.\n")
                    continue
            
            elif choice == '2':
                # Browse library
                print("\n📂 Opening folder browser...")
                selected = browse_directory(library_path)
                if selected:
                    path = Path(selected)
                    sample_files = list(path.rglob('*'))
                    audio_count = len([f for f in sample_files if f.suffix.lower() in AUDIO_EXTENSIONS])
                    print(f"\n📂 Selected: {path}")
                    print(f"🎵 Found ~{audio_count} audio files")
                    confirm = input("\nProceed with this directory? [Y/n]: ").strip().lower()
                    if confirm in ('', 'y', 'yes'):
                        return str(path)
                    else:
                        print("Cancelled. Let's try again.\n")
                        continue
                else:
                    print("\nBrowsing cancelled. Let's try again.\n")
                    continue
            
            elif choice == '3':
                break  # Fall through to manual path entry
            
            else:
                print("   ⚠️  Please enter 1, 2, or 3")
    else:
        if library_path:
            print(f"⚠️  Default library path no longer exists: {library_path}")
            print()
    
    # Manual path entry (original flow)
    print("Example paths:")
    print("  • /srv/.../Music/Sources/Clean/2Pac")
    print("  • /srv/.../Music/Sources/Clean/2Pac/Me Against the World")
    print("  • /srv/.../Music/Sources/Clean")
    print()
    
    while True:
        path_input = input("Enter the path to analyze (or 'q' to quit): ").strip()
        
        if path_input.lower() in ['q', 'quit', 'exit']:
            print("👋 Exiting...")
            sys.exit(0)
        
        # Expand ~ and handle quotes
        path_input = path_input.strip('\'"')
        path = Path(os.path.expanduser(path_input))
        
        if not path.exists():
            print(f"❌ Path does not exist: {path}")
            print("Please try again.\n")
            continue
        
        if not path.is_dir():
            print(f"❌ Path is not a directory: {path}")
            print("Please try again.\n")
            continue
        
        # Preview what will be scanned
        sample_files = list(path.rglob('*'))
        audio_count = len([f for f in sample_files if f.suffix.lower() in AUDIO_EXTENSIONS])
        
        print(f"\n📂 Directory: {path}")
        print(f"🎵 Found ~{audio_count} audio files")
        
        confirm = input("\nProceed with this directory? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return str(path)
        else:
            print("Cancelled. Let's try again.\n")


def get_int_input(prompt, default, min_val=None, max_val=None):
    """Get integer input with validation"""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"   ⚠️  Must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"   ⚠️  Must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("   ⚠️  Please enter a valid number")


def get_float_input(prompt, default, min_val=None, max_val=None):
    """Get float input with validation"""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"   ⚠️  Must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"   ⚠️  Must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("   ⚠️  Please enter a valid number")


def get_yes_no(prompt, default=True):
    """Get yes/no input"""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input in ['y', 'yes']


def configure_settings():
    """Interactive configuration"""
    config = Config()
    
    # Load saved settings
    saved = load_settings()
    config.default_library_path = saved.get('default_library_path')
    
    print("\n" + "=" * 70)
    print("⚙️  CONFIGURATION")
    print("=" * 70)
    print("\nPress Enter to accept defaults shown in [brackets]\n")
    
    # Default library path
    print("─" * 70)
    print("📚 DEFAULT LIBRARY PATH")
    print("   Set a default path to your music library for quick access")
    print("   This will be saved between runs")
    if config.default_library_path:
        print(f"   Current: {config.default_library_path}")
        print("   • 1 = Keep current (default)")
        print("   • 2 = Change path")
        print("   • 3 = Clear (remove default)")
        lib_choice = get_int_input("Library path", default=1, min_val=1, max_val=3)
        if lib_choice == 2:
            new_path = input("   Enter new library path: ").strip().strip('\'"')
            new_path = os.path.expanduser(new_path)
            if os.path.isdir(new_path):
                config.default_library_path = new_path
                saved['default_library_path'] = new_path
                save_settings(saved)
                print(f"   ✅ Library path saved: {new_path}")
            else:
                print(f"   ❌ Path does not exist: {new_path}")
                print("   Keeping previous setting.")
        elif lib_choice == 3:
            config.default_library_path = None
            saved.pop('default_library_path', None)
            save_settings(saved)
            print("   ✅ Library path cleared")
    else:
        set_lib = get_yes_no("Set a default library path?", default=False)
        if set_lib:
            new_path = input("   Enter library path: ").strip().strip('\'"')
            new_path = os.path.expanduser(new_path)
            if os.path.isdir(new_path):
                config.default_library_path = new_path
                saved['default_library_path'] = new_path
                save_settings(saved)
                print(f"   ✅ Library path saved: {new_path}")
            else:
                print(f"   ❌ Path does not exist: {new_path}")
    
    # Dry run mode
    print("\n" + "─" * 70)
    print("🧪 DRY RUN MODE")
    print("   Test mode - analyzes files but doesn't write tags")
    print("   Recommended: Enable for first run to see results")
    config.dry_run = get_yes_no("Enable dry run mode?", default=True)
    
    # Analysis mode
    print("\n" + "─" * 70)
    print("🎯 ANALYSIS MODE")
    print("   What to analyze and tag:")
    print("   • 1 = Genres & Moods (both)")
    print("   • 2 = Genres only")
    print("   • 3 = Moods only")
    mode_choice = get_int_input("Analysis mode", default=1, min_val=1, max_val=3)
    config.enable_genres = mode_choice in (1, 2)
    config.enable_moods = mode_choice in (1, 3)
    
    # Genre settings (only if genres enabled)
    if config.enable_genres:
        print("\n" + "─" * 70)
        print("🎸 GENRE SETTINGS")
        print("   How many genre tags to write per song")
        print("   Recommended: 2-4 genres")
        print("   • 1 = Only top genre")
        print("   • 3 = Balanced (good variety)")
        print("   • 5 = Comprehensive (may include less relevant)")
        config.top_n_genres = get_int_input("Number of genres to write", default=3, min_val=1, max_val=10)
        
        # Genre threshold
        print("\n   Genre confidence threshold (as percentage)")
        print("   Only include genres above this confidence level")
        print("   • 5% = Very inclusive (more genres)")
        print("   • 15% = Balanced (recommended)")
        print("   • 25% = Strict (fewer, higher confidence)")
        print("   • 35% = Very strict (may get 0-1 genres)")
        threshold_pct = get_float_input("Genre threshold (%)", default=15, min_val=1, max_val=50)
        config.genre_threshold = threshold_pct / 100.0
        
        # Genre formatting
        print("\n   Genre tag formatting")
        print("   How to format genre tags like 'Rock---Alternative Rock'")
        print("   • 1 = 'Rock - Alternative Rock' (parent - child)")
        print("   • 2 = 'Alternative Rock - Rock' (child - parent)")
        print("   • 3 = 'Alternative Rock' (child only)")
        print("   • 4 = 'Rock---Alternative Rock' (raw/no formatting)")
        format_choice = get_int_input("Genre format", default=1, min_val=1, max_val=4)
        format_map = {
            1: 'parent_child',
            2: 'child_parent',
            3: 'child_only',
            4: 'raw'
        }
        config.genre_format = format_map[format_choice]
    
    # Mood settings (only if moods enabled)
    if config.enable_moods:
        print("\n" + "─" * 70)
        print("😊 MOOD SETTINGS")
        print("   Mood confidence threshold (as percentage)")
        print("   Note: Mood predictions are typically MUCH lower confidence than genres")
        print("   Often in the 0.01% - 5% range!")
        print("   • 0.1% = Very inclusive (will get many moods)")
        print("   • 0.5% = Inclusive (recommended to start)")
        print("   • 1% = Balanced")
        print("   • 3% = Strict (may get few/no moods)")
        mood_threshold_pct = get_float_input("Mood threshold (%)", default=0.5, min_val=0.01, max_val=20)
        config.mood_threshold = mood_threshold_pct / 100.0
    
    # Write confidence scores
    print("\n" + "─" * 70)
    print("📊 CONFIDENCE SCORES")
    print("   Write confidence percentages to additional tags")
    print("   Example: 'ESSENTIA_GENRE: Alternative Rock: 32%, Indie Rock: 23%'")
    config.write_confidence_tags = get_yes_no("Write confidence score tags?", default=True)
    
    # Overwrite existing
    print("\n" + "─" * 70)
    print("♻️  EXISTING TAGS")
    print("   What to do if files already have existing tags")
    print("   • Overwrite: Replace existing tags")
    print("   • Skip: Leave files with existing tags untouched")
    config.overwrite_existing = get_yes_no("Overwrite existing tags?", default=False)
    
    # Verbose output
    print("\n" + "─" * 70)
    print("📢 VERBOSE OUTPUT")
    print("   Show detailed analysis info (top 10 predictions, etc.)")
    config.verbose = get_yes_no("Enable verbose output?", default=True)
    
    return config


def display_config_summary(config, music_path):
    """Display final configuration before processing"""
    print("\n" + "=" * 70)
    print("📋 FINAL SETTINGS")
    print("=" * 70)
    print(f"📂 Target directory: {music_path}")
    print(f"📁 Model directory: {MODEL_DIR}")
    if config.default_library_path:
        print(f"📚 Default library: {config.default_library_path}")
    
    if config.enable_genres and config.enable_moods:
        print(f"\n🎯 Analysis mode: Genres & Moods")
    elif config.enable_genres:
        print(f"\n🎯 Analysis mode: Genres only")
    else:
        print(f"\n🎯 Analysis mode: Moods only")
    
    if config.enable_genres:
        print(f"\n🎸 Genre Settings:")
        print(f"   • Number of genres: {config.top_n_genres}")
        print(f"   • Confidence threshold: {config.genre_threshold:.2%}")
        print(f"   • Format style: {config.genre_format}")
    if config.enable_moods:
        print(f"\n😊 Mood Settings:")
        print(f"   • Confidence threshold: {config.mood_threshold:.2%}")
    print(f"\n📊 Other Settings:")
    print(f"   • Dry run mode: {config.dry_run}")
    print(f"   • Write confidence tags: {config.write_confidence_tags}")
    print(f"   • Overwrite existing: {config.overwrite_existing}")
    print(f"   • Verbose output: {config.verbose}")
    
    if config.dry_run:
        print(f"\n⚠️  DRY RUN MODE - No files will be modified!")
    else:
        print(f"\n⚠️  LIVE MODE - Files WILL be modified!")
    
    # Set log file path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.log_file = f"essentia_tagger_{timestamp}.log"
    print(f"\n📝 Log file: {config.log_file}")
    
    print("=" * 70 + "\n")
    
    if not config.dry_run:
        confirm = input("Ready to proceed? [Y/n]: ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("Cancelled.")
            sys.exit(0)


def parse_arguments():
    """Parse command-line arguments for automated/non-interactive mode"""
    parser = argparse.ArgumentParser(
        description='Analyze music files with Essentia and write genre/mood tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (no arguments)
  python tag_music.py
  
  # Automated mode with path
  python tag_music.py /path/to/music --auto
  
  # Automated mode with custom settings
  python tag_music.py /path/to/music --auto --genres 4 --genre-threshold 20 --mood-threshold 1
  
  # Moods only (no genre tagging)
  python tag_music.py /path/to/music --auto --no-genres
  
  # Genres only (no mood tagging)
  python tag_music.py /path/to/music --auto --no-moods
  
  # Watch a specific file (for file watcher integration)
  python tag_music.py /path/to/song.flac --auto --single-file
  
  # Dry run to test
  python tag_music.py /path/to/music --auto --dry-run

Genre format styles:
  parent_child: "Rock - Alternative Rock" (default)
  child_parent: "Alternative Rock - Rock"
  child_only:   "Alternative Rock"
  raw:          "Rock---Alternative Rock"
"""
    )
    
    # Positional argument for path
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to music file or directory to analyze'
    )
    
    # Mode flags
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Run in automated (non-interactive) mode'
    )
    
    parser.add_argument(
        '--single-file', '-f',
        action='store_true',
        help='Process a single file instead of directory (for file watcher integration)'
    )
    
    # Genre settings
    parser.add_argument(
        '--genres', '-g',
        type=int,
        default=3,
        metavar='N',
        help='Number of genres to write (default: 3)'
    )
    
    parser.add_argument(
        '--genre-threshold', '-gt',
        type=float,
        default=15.0,
        metavar='PCT',
        help='Genre confidence threshold in percent (default: 15)'
    )
    
    parser.add_argument(
        '--genre-format', '-gf',
        choices=['parent_child', 'child_parent', 'child_only', 'raw'],
        default='parent_child',
        help='Genre tag format style (default: parent_child)'
    )
    
    # Analysis mode settings
    parser.add_argument(
        '--no-genres',
        action='store_true',
        help='Disable genre analysis (moods only)'
    )
    
    parser.add_argument(
        '--no-moods',
        action='store_true',
        help='Disable mood analysis (genres only)'
    )
    
    parser.add_argument(
        '--mood-threshold', '-mt',
        type=float,
        default=0.5,
        metavar='PCT',
        help='Mood confidence threshold in percent (default: 0.5)'
    )
    
    # Other settings
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Analyze files but do not write tags'
    )
    
    parser.add_argument(
        '--no-confidence-tags',
        action='store_true',
        help='Do not write confidence score tags'
    )
    
    parser.add_argument(
        '--overwrite', '-o',
        action='store_true',
        help='Overwrite existing genre tags'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (disable verbose mode)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Directory for log files (default: current directory)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Directory containing Essentia models (default: ~/essentia_models)'
    )
    
    parser.add_argument(
        '--library',
        type=str,
        default=None,
        metavar='DIR',
        help='Default music library path (saved for future runs)'
    )
    
    return parser.parse_args()


def config_from_args(args):
    """Create Config object from command-line arguments"""
    if args.no_genres and args.no_moods:
        print("❌ Error: Cannot disable both genres and moods")
        print("   Use --no-genres OR --no-moods, not both")
        sys.exit(1)
    
    config = Config()
    config.dry_run = args.dry_run
    config.enable_genres = not args.no_genres
    config.enable_moods = not args.no_moods
    config.top_n_genres = args.genres
    config.genre_threshold = args.genre_threshold / 100.0
    config.mood_threshold = args.mood_threshold / 100.0
    config.write_confidence_tags = not args.no_confidence_tags
    config.overwrite_existing = args.overwrite
    config.verbose = not args.quiet
    config.genre_format = args.genre_format
    
    # Handle library path
    if args.library:
        lib_path = os.path.expanduser(args.library)
        if os.path.isdir(lib_path):
            saved = load_settings()
            saved['default_library_path'] = lib_path
            save_settings(saved)
            config.default_library_path = lib_path
    
    # Set up log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"essentia_tagger_{timestamp}.log"
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        config.log_file = str(log_dir / log_filename)
    else:
        config.log_file = log_filename
    
    return config


def process_single_file(filepath, analyzer, tag_writer, config, logger):
    """Process a single audio file (for file watcher integration)"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.log(f"❌ File not found: {filepath}")
        return False
    
    if filepath.suffix.lower() not in AUDIO_EXTENSIONS:
        logger.log(f"⏭️ Skipping non-audio file: {filepath}")
        return False
    
    logger.log(f"🎵 Processing: {filepath.name}")
    
    results = analyzer.analyze_file(filepath)
    
    if results:
        # Print results
        if config.enable_genres:
            if results.get('genres'):
                genre_list = [f"{g['label']} ({g['confidence']:.1%})" for g in results['genres']]
                logger.log(f"   🎸 Genres: {', '.join(genre_list)}")
            if results.get('formatted_genres'):
                logger.log(f"   🎸 Formatted: {', '.join(results['formatted_genres'])}")
        
        if config.enable_moods and results.get('moods'):
            mood_list = [f"{m['label']} ({m['confidence']:.1%})" for m in results['moods'][:3]]
            logger.log(f"   😊 Moods: {', '.join(mood_list)}")
        
        # Log to file
        logger.log_analysis(filepath, results, filepath.name)
        
        # Write tags
        tag_writer.write_tags(filepath, results)
        
        if not config.dry_run:
            logger.log("   ✅ Tags written")
        
        return True
    else:
        logger.log(f"   ❌ Analysis failed")
        return False


def main():
    """Main entry point"""
    args = parse_arguments()
    logger = None
    
    # Check if we should run in automated mode
    if args.auto or args.single_file:
        # Automated/CLI mode
        if not args.path:
            print("❌ Error: Path is required in automated mode")
            print("   Use: python tag_music.py /path/to/music --auto")
            sys.exit(1)
        
        music_path = os.path.expanduser(args.path)
        
        # Update model directory if specified
        global MODEL_DIR, EMBEDDING_MODEL, GENRE_MODEL, GENRE_METADATA, MOOD_MODEL, MOOD_METADATA
        if args.model_dir:
            MODEL_DIR = os.path.expanduser(args.model_dir)
            EMBEDDING_MODEL = f"{MODEL_DIR}/discogs-effnet-bs64-1.pb"
            GENRE_MODEL = f"{MODEL_DIR}/genre_discogs400-discogs-effnet-1.pb"
            GENRE_METADATA = f"{MODEL_DIR}/genre_discogs400-discogs-effnet-1.json"
            MOOD_MODEL = f"{MODEL_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
            MOOD_METADATA = f"{MODEL_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.json"
        
        config = config_from_args(args)
        
        try:
            logger = Logger(config.log_file)
            logger.log_config(config, music_path)
            
            # Summary output
            mode_str = "DRY RUN" if config.dry_run else "LIVE"
            logger.log(f"🎸 Essentia Tagger [{mode_str}]")
            logger.log(f"   Path: {music_path}")
            if config.enable_genres:
                logger.log(f"   Genres: {config.top_n_genres} (threshold: {config.genre_threshold:.1%})")
            if config.enable_moods:
                logger.log(f"   Moods: enabled (threshold: {config.mood_threshold:.2%})")
            logger.log("")
            
            analyzer = EssentiaAnalyzer(config, logger)
            tag_writer = TagWriter(config, logger)
            
            if args.single_file:
                # Single file mode
                success = process_single_file(music_path, analyzer, tag_writer, config, logger)
                sys.exit(0 if success else 1)
            else:
                # Directory mode
                if not os.path.isdir(music_path):
                    logger.log(f"❌ Error: Not a directory: {music_path}")
                    sys.exit(1)
                
                scan_library(music_path, analyzer, tag_writer, config, logger)
            
            logger.log("\n✅ Processing complete!")
            logger.log(f"📝 Log: {config.log_file}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        finally:
            if logger:
                logger.close()
    
    else:
        # Interactive mode (original behavior)
        try:
            # Load saved settings for library path
            saved = load_settings()
            
            # Create a temporary config to hold library path for get_music_path
            temp_config = Config()
            temp_config.default_library_path = saved.get('default_library_path')
            
            # Get path from user
            music_path = get_music_path(temp_config)
            
            # Configure settings interactively
            config = configure_settings()
            
            # Show summary and confirm
            display_config_summary(config, music_path)
            
            # Initialize logger
            logger = Logger(config.log_file)
            logger.log_config(config, music_path)
            
            # Initialize
            analyzer = EssentiaAnalyzer(config, logger)
            tag_writer = TagWriter(config, logger)
            
            # Process library
            scan_library(music_path, analyzer, tag_writer, config, logger)
            
            logger.log("\n" + "=" * 70)
            logger.log("✅ PROCESSING COMPLETE!")
            logger.log("=" * 70)
            logger.log(f"\n📝 Full log saved to: {config.log_file}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            sys.exit(1)
        finally:
            if logger:
                logger.close()


if __name__ == '__main__':
    main()
