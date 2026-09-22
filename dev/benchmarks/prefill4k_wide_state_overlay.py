#!/usr/bin/env python3
"""Grant the continuation oracle read-only friendship in a private header only."""
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build/prefill4k-wide")
    args = parser.parse_args()
    build = Path(args.build)
    source = build / "source/runtime/flash/FlashForward.hpp"
    original = source.read_text()
    needle = "private:\n  friend class FlashBatchForward;"
    if original.count(needle) != 1:
        raise SystemExit("private Forward friendship anchor changed; inspect before regenerating")
    patched = original.replace(needle, "private:\n  friend class FlashDeepPrefixOracle;\n  friend class FlashBatchForward;")
    destination = build / "state-source/runtime/flash/FlashForward.hpp"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_text() != patched:
        destination.write_text(patched)
    manifest = {
        "schema": "splash-private-wide-state-access-v1",
        "source": str(source),
        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "destination": str(destination),
        "destination_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "production_files_changed": False,
        "read_only_test_friend": "FlashDeepPrefixOracle",
    }
    (build / "state-source/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
