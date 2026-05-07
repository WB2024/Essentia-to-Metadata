"""Unit tests for riff_info.py.

All tests use in-memory synthesised WAV bytes where possible; real-file
tests use tempfile so no permanent files are created.
"""
import struct
import tempfile
import unittest
from pathlib import Path

from riff_info import (
    _build_list_info,
    _encode_subchunk,
    _locate_list_info,
    _parse_riff_info,
    read_riff_info,
    update_riff_info,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic WAV data
# ---------------------------------------------------------------------------

def _fmt_chunk():
    """Minimal 16-byte PCM fmt sub-chunk body."""
    # audio_format=1 (PCM), channels=1, sample_rate=44100,
    # byte_rate=88200, block_align=2, bits=16
    return struct.pack('<HHIIHH', 1, 1, 44100, 88200, 2, 16)


def _make_wav(*extra_chunks):
    """Return a minimal RIFF WAVE file as bytes.

    extra_chunks: sequence of raw bytes (complete chunks, header included)
    that will be appended after the fmt  and data chunks.
    """
    fmt_body = _fmt_chunk()
    fmt_chunk = b'fmt ' + struct.pack('<I', len(fmt_body)) + fmt_body

    samples = b'\x00\x00'       # one silent 16-bit sample
    data_chunk = b'data' + struct.pack('<I', len(samples)) + samples

    body = b'WAVE' + fmt_chunk + data_chunk
    for chunk in extra_chunks:
        body += chunk

    return b'RIFF' + struct.pack('<I', len(body)) + body


def _make_list_info(**kwargs):
    """Return a LIST/INFO chunk bytes with the given FOURCC=text items."""
    info_dict = {k.encode('ascii'): v for k, v in kwargs.items()}
    return _build_list_info(info_dict)


# ---------------------------------------------------------------------------
# _encode_subchunk
# ---------------------------------------------------------------------------

class TestEncodeSubchunk(unittest.TestCase):

    def test_even_length(self):
        # "Hi" + null = 3 bytes (odd) → padded to 4 bytes on disk; size=3
        chunk = _encode_subchunk(b'ICMT', 'Hi')
        fourcc = chunk[:4]
        size = struct.unpack_from('<I', chunk, 4)[0]
        value = chunk[8:8 + size]
        self.assertEqual(fourcc, b'ICMT')
        self.assertEqual(value, b'Hi\x00')
        # total on-disk length must be even
        self.assertEqual(len(chunk) % 2, 0)

    def test_odd_length_no_pad_needed(self):
        # "Yes" + null = 4 bytes (even) → no extra pad; size=4
        chunk = _encode_subchunk(b'ICMT', 'Yes')
        size = struct.unpack_from('<I', chunk, 4)[0]
        self.assertEqual(size, 4)
        self.assertEqual(len(chunk), 12)   # 4+4+4

    def test_empty_string(self):
        chunk = _encode_subchunk(b'ICMT', '')
        size = struct.unpack_from('<I', chunk, 4)[0]
        # just a null byte; size=1 (odd) → padded to 2 on disk
        self.assertEqual(size, 1)
        self.assertEqual(len(chunk) % 2, 0)

    def test_bytes_key(self):
        chunk = _encode_subchunk(b'IGNR', 'Rock')
        self.assertEqual(chunk[:4], b'IGNR')


# ---------------------------------------------------------------------------
# _parse_riff_info
# ---------------------------------------------------------------------------

class TestParseRiffInfo(unittest.TestCase):

    def test_no_list_info_returns_empty(self):
        wav = _make_wav()
        self.assertEqual(_parse_riff_info(wav), {})

    def test_single_key(self):
        list_chunk = _make_list_info(ICMT='Happy vibes')
        wav = _make_wav(list_chunk)
        result = _parse_riff_info(wav)
        self.assertEqual(result.get(b'ICMT'), 'Happy vibes')

    def test_multiple_keys(self):
        list_chunk = _make_list_info(ICMT='A comment', IGNR='Electronic')
        wav = _make_wav(list_chunk)
        result = _parse_riff_info(wav)
        self.assertEqual(result[b'ICMT'], 'A comment')
        self.assertEqual(result[b'IGNR'], 'Electronic')

    def test_odd_length_value_parses_correctly(self):
        # "Hi" is 2 chars → null-terminated is 3 bytes (odd)
        list_chunk = _make_list_info(ICMT='Hi')
        wav = _make_wav(list_chunk)
        result = _parse_riff_info(wav)
        self.assertEqual(result[b'ICMT'], 'Hi')

    def test_value_with_embedded_null_trimmed(self):
        # Manually craft a subchunk with an embedded null
        value = b'Foo\x00Bar\x00'
        sub = b'ICMT' + struct.pack('<I', len(value)) + value
        info_body = b'INFO' + sub
        list_chunk = b'LIST' + struct.pack('<I', len(info_body)) + info_body
        wav = _make_wav(list_chunk)
        result = _parse_riff_info(wav)
        self.assertEqual(result[b'ICMT'], 'Foo')

    def test_chunk_after_list_info_not_required(self):
        # LIST INFO at the very end of file should still parse
        list_chunk = _make_list_info(ICMT='end')
        wav = _make_wav(list_chunk)
        result = _parse_riff_info(wav)
        self.assertEqual(result[b'ICMT'], 'end')

    def test_non_info_list_chunk_ignored(self):
        # A LIST chunk with type 'adtl' should not be mistaken for LIST INFO
        adtl_body = b'adtl' + b'\x00' * 4
        adtl_chunk = b'LIST' + struct.pack('<I', len(adtl_body)) + adtl_body
        list_info = _make_list_info(ICMT='found me')
        wav = _make_wav(adtl_chunk, list_info)
        result = _parse_riff_info(wav)
        self.assertEqual(result[b'ICMT'], 'found me')


# ---------------------------------------------------------------------------
# _locate_list_info
# ---------------------------------------------------------------------------

class TestLocateListInfo(unittest.TestCase):

    def test_returns_none_when_absent(self):
        wav = _make_wav()
        self.assertIsNone(_locate_list_info(wav))

    def test_returns_offsets_when_present(self):
        list_chunk = _make_list_info(ICMT='x')
        wav = _make_wav(list_chunk)
        loc = _locate_list_info(wav)
        self.assertIsNotNone(loc)
        start, end = loc
        # The LIST/INFO chunk starts after fmt  and data chunks
        self.assertGreater(start, 12)
        self.assertGreater(end, start)
        # The bytes at start should be LIST
        self.assertEqual(wav[start:start + 4], b'LIST')


# ---------------------------------------------------------------------------
# read_riff_info
# ---------------------------------------------------------------------------

class TestReadRiffInfo(unittest.TestCase):

    def _write_temp(self, wav_bytes):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(wav_bytes)
        tmp.close()
        return tmp.name

    def test_read_no_info(self):
        path = self._write_temp(_make_wav())
        self.assertEqual(read_riff_info(path), {})

    def test_read_with_info(self):
        path = self._write_temp(_make_wav(_make_list_info(ICMT='Hello')))
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'Hello')

    def test_raises_for_non_wav(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(b'this is not a wav file at all')
        tmp.close()
        with self.assertRaises(ValueError):
            read_riff_info(tmp.name)


# ---------------------------------------------------------------------------
# update_riff_info
# ---------------------------------------------------------------------------

class TestUpdateRiffInfo(unittest.TestCase):

    def _write_temp(self, wav_bytes):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(wav_bytes)
        tmp.close()
        return tmp.name

    def test_insert_into_file_with_no_list_info(self):
        path = self._write_temp(_make_wav())
        update_riff_info(path, {b'ICMT': 'mood here'})
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'mood here')

    def test_update_existing_key(self):
        path = self._write_temp(_make_wav(_make_list_info(ICMT='old')))
        update_riff_info(path, {b'ICMT': 'new'})
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'new')

    def test_preserves_unmodified_keys(self):
        path = self._write_temp(_make_wav(_make_list_info(ICMT='comment', IGNR='Jazz')))
        update_riff_info(path, {b'ICMT': 'updated comment'})
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'updated comment')
        self.assertEqual(result[b'IGNR'], 'Jazz')

    def test_adds_new_key_alongside_existing(self):
        path = self._write_temp(_make_wav(_make_list_info(IGNR='Rock')))
        update_riff_info(path, {b'ICMT': 'new comment'})
        result = read_riff_info(path)
        self.assertEqual(result[b'IGNR'], 'Rock')
        self.assertEqual(result[b'ICMT'], 'new comment')

    def test_str_key_accepted(self):
        path = self._write_temp(_make_wav())
        update_riff_info(path, {'ICMT': 'str key works'})
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'str key works')

    def test_idempotent(self):
        path = self._write_temp(_make_wav())
        update_riff_info(path, {b'ICMT': 'same'})
        update_riff_info(path, {b'ICMT': 'same'})
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'same')
        # ICMT should not have been duplicated
        data = Path(path).read_bytes()
        self.assertEqual(data.count(b'ICMT'), 1)

    def test_riff_size_header_is_correct_after_insert(self):
        path = self._write_temp(_make_wav())
        update_riff_info(path, {b'ICMT': 'test'})
        data = Path(path).read_bytes()
        stored_size = struct.unpack_from('<I', data, 4)[0]
        self.assertEqual(stored_size, len(data) - 8)

    def test_riff_size_header_is_correct_after_update(self):
        path = self._write_temp(_make_wav(_make_list_info(ICMT='short')))
        update_riff_info(path, {b'ICMT': 'a much longer comment than before'})
        data = Path(path).read_bytes()
        stored_size = struct.unpack_from('<I', data, 4)[0]
        self.assertEqual(stored_size, len(data) - 8)

    def test_fmt_chunk_preserved(self):
        # The fmt  chunk must survive a write
        wav = _make_wav()
        path = self._write_temp(wav)
        update_riff_info(path, {b'ICMT': 'check'})
        data = Path(path).read_bytes()
        self.assertIn(b'fmt ', data)
        self.assertIn(b'data', data)

    def test_odd_value_round_trips(self):
        path = self._write_temp(_make_wav())
        update_riff_info(path, {b'ICMT': 'Hi'})   # 2 chars → odd null-terminated
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], 'Hi')

    def test_empty_value_round_trips(self):
        path = self._write_temp(_make_wav())
        update_riff_info(path, {b'ICMT': ''})
        result = read_riff_info(path)
        self.assertEqual(result[b'ICMT'], '')

    def test_raises_for_non_wav(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(b'definitely not a wav')
        tmp.close()
        with self.assertRaises(ValueError):
            update_riff_info(tmp.name, {b'ICMT': 'x'})

    def test_mood_marker_merge_via_comment_merge(self):
        """Integration: merging a [MOOD:] marker into an existing ICMT value."""
        from comment_merge import build_mood_marker, merge_mood_into_comment
        path = self._write_temp(_make_wav(_make_list_info(ICMT='Energy 7 - 8A')))
        existing = read_riff_info(path)[b'ICMT']
        marker = build_mood_marker(['Happy', 'Energetic'])
        new_val = merge_mood_into_comment(existing, marker)
        update_riff_info(path, {b'ICMT': new_val})
        result = read_riff_info(path)[b'ICMT']
        self.assertIn('[MOOD:', result)
        self.assertIn('Energy 7 - 8A', result)

    def test_mood_marker_idempotent_via_comment_merge(self):
        """Re-running the merge should not accumulate markers."""
        from comment_merge import build_mood_marker, merge_mood_into_comment
        path = self._write_temp(_make_wav())
        marker = build_mood_marker(['Happy'])
        for _ in range(3):
            existing = read_riff_info(path).get(b'ICMT', '')
            new_val = merge_mood_into_comment(existing, marker)
            update_riff_info(path, {b'ICMT': new_val})
        result = read_riff_info(path)[b'ICMT']
        self.assertEqual(result.count('[MOOD:'), 1)


if __name__ == '__main__':
    unittest.main()
