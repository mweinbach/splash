#!/usr/bin/env python3
"""Root-only replay of the frozen row-pair primitive on qualified live inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / 'build/raw-q4-rowpair-sep22-component-v5'
CAPTURE = ROOT / 'build/release/flash/sep22-raw-q4-current-layer1-input-v1/layer1-QKV-VerifyR4.json'
PROOF = ROOT / 'build/release/flash/sep22-raw-q4-current-layer1-capture-proof-v1.json'
REPORT = ROOT / 'build/release/flash/sep22-raw-q4-rowpair-layer1-current-v1.json'
PINS = {
    'oracle': '12de25787566461bf4f627406cd945473712d104c0c55bf2133fa80a7a518abf',
    'component.metallib': 'ee54f8f2256c5ae9b840d71e647c8329415494f466e200a66dc0ec5a63e1cdd9',
    'manifest.json': '73315e1522db6a89968fcc7556e81678dcdad07c966809a40f141de011af416a',
}


def main():
    for name, pin in PINS.items():
        if hashlib.sha256((BUILD / name).read_bytes()).hexdigest() != pin:
            raise RuntimeError(f'Frozen rowpair component drift: {name}')
    capture = json.loads(CAPTURE.read_text())
    proof = json.loads(PROOF.read_text())
    if capture != proof:
        raise RuntimeError('Actual capture metadata and completed preservation proof differ')
    if (capture.get('pass') is not True or
            capture.get('current_input_capture_preservation_proved') is not True or
            capture.get('actual_MTP_proposal_capture') is not False or
            capture.get('payload_disk_bytes') != 20480 or
            capture.get('actual_capture_owned_buffer_guard_cases') != 6 or
            capture.get('actual_capture_clone_executable_sha256') !=
            '7a8d9732776f3f53080053a4d9ecf098411649c04e68439a552ac06ad8af9e3d' or
            capture.get('actual_capture_library_sha256') !=
            '06ccaac045fadf6122bef528544f866d81e000ffdf5fd8a5b58a529e3a212d6b' or
            capture['preservation']['all_bytes_equal'] is not True or
            capture['preservation']['full_frames_compared'] != 3 or
            capture['allocation']['backend_destroyed'] is not True):
        raise RuntimeError('Completed actual current-input capture qualification required')
    for suffix in ('', '.partial', '.failure.json', '.writing'):
        if Path(str(REPORT) + suffix).exists():
            raise RuntimeError('Fresh actual-input component report required')
    return subprocess.run([str(BUILD / 'oracle'), '--gpu',
                           str(BUILD / 'component.metallib'),
                           str(ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'),
                           str(REPORT), str(CAPTURE)], cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
