#!/usr/bin/env python3
"""Root-only bounded synthetic projection with independent artifact pins."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / 'build/raw-q4-rowpair-sep22-component-v5'
PINS = {
    'oracle': '12de25787566461bf4f627406cd945473712d104c0c55bf2133fa80a7a518abf',
    'component.metallib': 'ee54f8f2256c5ae9b840d71e647c8329415494f466e200a66dc0ec5a63e1cdd9',
    'manifest.json': '73315e1522db6a89968fcc7556e81678dcdad07c966809a40f141de011af416a',
    'run-root-synthetic.sh': '811a5a4f5bad2b3fa4381c3fb87777870aa09ff653902e38d0d0d82d21fe9d28',
}
REPORT = ROOT / 'build/release/flash/sep22-raw-q4-rowpair-layer1-synthetic-v1.json'


def main():
    for name, expected in PINS.items():
        if hashlib.sha256((BUILD / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen synthetic component drift: {name}')
    for suffix in ('', '.partial', '.failure.json', '.writing'):
        if Path(str(REPORT) + suffix).exists():
            raise RuntimeError('Fresh synthetic proof report required')
    return subprocess.run(['sh', str(BUILD / 'run-root-synthetic.sh')], cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
