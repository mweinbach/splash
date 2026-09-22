#!/usr/bin/env python3
"""Copy a handed-off allrow source tree into an isolated rollback build, metadata only."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-build", type=Path, default=ROOT / "build/prefill4k-allrows-full512")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-allrows-rollback")
    args = parser.parse_args()
    source, output = args.source_build.resolve(), args.output.resolve()
    if ROOT / "build" not in output.parents or output == source:
        raise ValueError("Rollback build must be a different private directory beneath build")
    original_manifest = json.loads((source / "overlay-manifest.json").read_text())
    manifest = dict(original_manifest)
    manifest.update({
        "rollback_source_build": str(source),
        "rollback_read_only_friend": "FlashAllrowsRollbackOracle",
        "rollback_payload_bytes_read": 0,
        "normal_sources_modified": False,
        "files": [],
    })
    for record in original_manifest["files"]:
        relative = record["path"]
        original = (source / "source" / relative).read_bytes()
        if hashlib.sha256(original).hexdigest() != record["overlay_sha256"]:
            raise ValueError(f"Source handoff is not fresh: {relative}")
        text = original.decode()
        if relative == "runtime/flash/FlashForward.hpp":
            request = "  friend class FlashDeepPrefixOracle;"
            forward = "private:\n  friend class FlashBatchForward;"
            if text.count(request) != 1 or text.count(forward) != 1:
                raise ValueError("Request/Forward friendship anchors changed")
            text = text.replace(request, request + "\n  friend class FlashAllrowsRollbackOracle;")
            text = text.replace(forward, "private:\n  friend class FlashDeepPrefixOracle;\n  friend class FlashAllrowsRollbackOracle;\n  friend class FlashBatchForward;")
        modified = text.encode()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_bytes() != modified:
            destination.write_bytes(modified)
        item = dict(record)
        item["rollback_input_sha256"] = record["overlay_sha256"]
        item["overlay_sha256"] = hashlib.sha256(modified).hexdigest()
        item["patched"] = item["patched"] or modified != original
        manifest["files"].append(item)
    path = output / "overlay-manifest.json"
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    print(json.dumps({"prepared": str(output), "source_build": str(source), "files": len(manifest["files"]),
                      "gpu_executed": False, "payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
