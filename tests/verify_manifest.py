"""Verify the runtime payload against its per-file SHA-256 manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
count = 0
size = 0
for item in data['files']:
    p = ROOT / item['path']
    assert p.resolve().is_relative_to(ROOT), item['path']
    assert p.is_file(), item['path']
    assert p.stat().st_size == item['bytes'], item['path']
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == item['sha256'], item['path']
    count += 1
    size += item['bytes']
assert size == data['payload_bytes']
print('RUNTIME_FILES_VERIFIED', count, 'BYTES', size, flush=True)
