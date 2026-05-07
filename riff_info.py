"""Read and write RIFF INFO (LIST/INFO) chunks in WAV files.

RIFF/WAV structure:
  b'RIFF' <4:LE-uint32 size>  b'WAVE'
    chunk ...
    b'LIST' <4:LE-uint32 size>  b'INFO'
      b'ICMT' <4:LE-uint32 sub-size>  <null-terminated text, padded to even>
      ...
    chunk ...

The RIFF size field = total_file_size - 8 (excludes the 8-byte RIFF header).
INFO sub-chunk size = length of data including null terminator.
If that length is odd a single silent padding byte follows on disk (not
counted in the size field).

Encoding: RIFF INFO conventionally uses Windows Latin-1 (cp1252).  ASCII
mood markers and genre strings encode correctly; non-ASCII characters in
existing tags are decoded with replacement rather than raising an error.
"""
import struct
from pathlib import Path

_RIFF = b'RIFF'
_WAVE = b'WAVE'
_LIST = b'LIST'
_INFO = b'INFO'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_riff_wave(data, label=''):
    if len(data) < 12 or data[:4] != _RIFF or data[8:12] != _WAVE:
        raise ValueError(f"Not a RIFF WAVE file{': ' + label if label else ''}")


def _parse_riff_info(data):
    """Return {fourcc_bytes: str} from the first LIST/INFO chunk in *data*.

    *data* must already be validated as a RIFF WAVE file.
    Returns an empty dict when no LIST/INFO chunk is present.
    """
    pos = 12
    while pos + 8 <= len(data):
        fourcc = data[pos:pos + 4]
        size = struct.unpack_from('<I', data, pos + 4)[0]
        content_start = pos + 8
        content_end = content_start + size

        if fourcc == _LIST and data[content_start:content_start + 4] == _INFO:
            result = {}
            sub_pos = content_start + 4
            while sub_pos + 8 <= content_end:
                sub_fourcc = data[sub_pos:sub_pos + 4]
                sub_size = struct.unpack_from('<I', data, sub_pos + 4)[0]
                raw = data[sub_pos + 8:sub_pos + 8 + sub_size]
                null_idx = raw.find(b'\x00')
                if null_idx >= 0:
                    raw = raw[:null_idx]
                result[sub_fourcc] = raw.decode('latin-1', errors='replace')
                # advance past subchunk + padding
                sub_pos += 8 + sub_size + (sub_size % 2)
            return result

        # advance past chunk + padding
        pos += 8 + size + (size % 2)

    return {}


def _encode_subchunk(fourcc, text):
    """Encode one INFO sub-chunk: FOURCC + LE-size + null-terminated value + pad."""
    value = text.encode('latin-1', errors='replace') + b'\x00'
    chunk = fourcc + struct.pack('<I', len(value)) + value
    if len(value) % 2 != 0:
        chunk += b'\x00'    # silent pad byte; not reflected in size
    return chunk


def _build_list_info(info_dict):
    """Build a complete LIST/INFO chunk from {fourcc_bytes: str}."""
    body = _INFO + b''.join(_encode_subchunk(k, v) for k, v in info_dict.items())
    return _LIST + struct.pack('<I', len(body)) + body


def _locate_list_info(data):
    """Return (start, end) byte offsets of the first LIST/INFO chunk, or None."""
    pos = 12
    while pos + 8 <= len(data):
        fourcc = data[pos:pos + 4]
        size = struct.unpack_from('<I', data, pos + 4)[0]
        content_start = pos + 8
        pad = size % 2
        end = content_start + size + pad

        if fourcc == _LIST and data[content_start:content_start + 4] == _INFO:
            return pos, end

        pos = end

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_riff_info(filepath):
    """Return {fourcc_bytes: str} for every sub-chunk in the WAV's LIST/INFO block.

    Returns an empty dict when no LIST/INFO chunk is present.
    Raises ValueError when the file is not a RIFF WAVE.
    """
    data = Path(filepath).read_bytes()
    _validate_riff_wave(data, str(filepath))
    return _parse_riff_info(data)


def update_riff_info(filepath, updates):
    """Update or insert RIFF INFO sub-chunks in a WAV file in-place.

    *updates* maps FOURCC keys (bytes or 4-char str) to str values.
    Existing INFO sub-chunks not present in *updates* are preserved unchanged.
    Raises ValueError when the file is not a RIFF WAVE.
    """
    path = Path(filepath)
    data = path.read_bytes()
    _validate_riff_wave(data, str(filepath))

    # Normalise key types to bytes
    norm = {(k.encode('ascii') if isinstance(k, str) else k): v
            for k, v in updates.items()}

    # Merge: existing keys not in updates are preserved
    merged = dict(_parse_riff_info(data))
    merged.update(norm)

    new_list = _build_list_info(merged)

    loc = _locate_list_info(data)
    if loc is not None:
        start, end = loc
        new_data = data[:start] + new_list + data[end:]
    else:
        new_data = data + new_list

    # Fix RIFF size header (excludes the 8-byte 'RIFF'+size header itself)
    riff_size = len(new_data) - 8
    new_data = new_data[:4] + struct.pack('<I', riff_size) + new_data[8:]

    path.write_bytes(new_data)
