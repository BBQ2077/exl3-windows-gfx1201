"""Build the release: refresh manifest.json, then write the A pack (python/) and the B pack (everything else).

    python\\python.exe -I tools\\build_release.py <out_dir> <version> [A|B]

Runtime-generated data (cache/, tmp/, logs/, state/, __pycache__/), the compile tree build/, *.log, *.prev and
*.bak* backups are never packed or listed in the manifest.
models.json ships in the B pack (default GGUFRun layout) but is user-editable, so it is not hashed."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MUTABLE = ['models.json', 'cache/', 'tmp/', 'logs/', 'state/', '**/__pycache__/']
SKIP_DIRS = {'cache', 'tmp', 'logs', 'state', 'build', '.git'}
NAME = 'EXL3-gfx1201'


def payload():
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if (not p.is_file() or rel.parts[0] in SKIP_DIRS or '__pycache__' in rel.parts or p.suffix in ('.pyc', '.prev', '.log')
                or '.bak' in p.name):
            continue
        yield rel.as_posix(), p


def sha256(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    out, version = Path(sys.argv[1]).resolve(), sys.argv[2]
    only = sys.argv[3] if len(sys.argv) > 3 else 'AB'
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for rel, p in payload():
        if rel in ('manifest.json', 'models.json'):
            continue
        files.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    manifest = {'format': 1, 'version': version,
                'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                'payload_bytes': sum(f['bytes'] for f in files), 'mutable_paths': MUTABLE, 'files': files}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('MANIFEST', len(files), 'files', manifest['payload_bytes'], 'bytes', flush=True)

    packs = {'A': out / f'{NAME}-A-python-{version}.zip', 'B': out / f'{NAME}-B-app-{version}.zip'}
    packs = {k: v for k, v in packs.items() if k in only}
    zips = {k: zipfile.ZipFile(v, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) for k, v in packs.items()}
    counts = {'A': 0, 'B': 0}
    try:
        for rel, p in payload():
            k = 'A' if rel.startswith('python/') else 'B'
            if k not in zips:
                continue
            zips[k].write(p, rel)
            counts[k] += 1
    finally:
        for z in zips.values():
            z.close()
    for k, p in packs.items():
        print(f'PACK {k} {p.name} files={counts[k]} bytes={p.stat().st_size} sha256={sha256(p)}', flush=True)


if __name__ == '__main__':
    main()
