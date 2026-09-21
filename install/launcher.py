#!/usr/bin/env python3
"""Serve in the foreground, or connect an installed agent to the local server.

An optional .splash-local-profile.json supplies runtime environment defaults
for its qualified local model and hardware. Explicit SPLASH_FLASH_* values
take precedence. Missing, unknown or mismatched profiles leave normal defaults
in place; profiles never change compiler or build settings. Disabling a parent
route also disables its implied dependent defaults. Explicit dependent values
remain unchanged so the runtime can report a contradictory configuration.
"""

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

try:
    from . import catalog, clients, paths
    from . import models as model_artifacts
except ImportError:  # Executed directly by the source or packaged entry point.
    import catalog
    import clients
    import models as model_artifacts
    import paths

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
LOCAL_SCHEMA = "splash-local-qwen4-affine-v1"
LOCAL_PROFILE = {
    "schema_version": 1,
    "profile": "m5-ultra-flash-next-v10",
    "source_identity_sha256": "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
    "architecture": "qwen4_exp",
    "cpu_brand": "Apple M5 Ultra",
    "minimum_physical_ram_bytes": 256 * 1024**3,
    "environment": {
        "SPLASH_FLASH_QMV_F32": "1",
        "SPLASH_FLASH_FUSE_HC": "1",
        "SPLASH_FLASH_FUSE_GDN": "1",
        "SPLASH_FLASH_QSA_F32": "1",
        "SPLASH_FLASH_DENSE_CACHE": "1",
        "SPLASH_FLASH_BLOCKED_MOE": "1",
        "SPLASH_FLASH_PREFILL_ROWS": "2048",
        "SPLASH_FLASH_BATCH": "1",
        "SPLASH_FLASH_MTP": "1",
        "SPLASH_FLASH_EXPERT_QMV": "1",
        "SPLASH_FLASH_GDN_STAGED": "1",
        "SPLASH_FLASH_QSA_MPP": "1",
        "SPLASH_FLASH_MTP_QSA_F32": "1",
        "SPLASH_FLASH_MTP_QSA_MPP": "1",
        "SPLASH_FLASH_FLOAT_DENSE_CACHE": "1",
        "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE": "1",
        "SPLASH_FLASH_MOE_Q4X8": "1",
        "SPLASH_FLASH_MOE_M64": "1",
        "SPLASH_FLASH_BATCH_MTP": "1",
        "SPLASH_FLASH_BATCH_PREFILL": "1",
        "SPLASH_FLASH_BATCH_MTP_PREFILL": "1",
        "SPLASH_FLASH_BATCH_PREFILL_ROWS": "2048",
        "SPLASH_FLASH_MTP_DRAFT_DEPTH": "15",
        "SPLASH_FLASH_PLE_LOOKUP_FUSED": "1",
        "SPLASH_FLASH_PLE_POST_FUSED": "1",
        "SPLASH_FLASH_GPU_GREEDY": "1",
        "SPLASH_FLASH_INT8_HEAD": "1",
        "SPLASH_FLASH_MOE_DIRECT_A": "1",
        "SPLASH_FLASH_QSA_ROW_TILES": "1",
        "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT": "1",
        "SPLASH_FLASH_SHARED_EXPERT_FUSED": "1",
        "SPLASH_FLASH_DENSE_M64_OUT": "1",
        "SPLASH_FLASH_GDN_BATCH_ILP": "1",
        "SPLASH_FLASH_MTP_QMV_F32": "1",
        "SPLASH_FLASH_HC_UP_F32_MPP": "1",
        "SPLASH_FLASH_GDN_LAZY_ROLLBACK": "1",
        "SPLASH_FLASH_MTP_Q8_BF16_REGISTER": "1",
        "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "1",
        "SPLASH_FLASH_PLE_SSD_STREAMING": "1",
    },
}

# Optional artifacts are selected dynamically after the local profile passes
# its source/model/hardware gate. Static defaults never force a missing path.
LOCAL_SAVED_OPERAND_QUALIFICATION = {
    "relative_path": "install/local-models/Flash-Next-operands-v1",
    "manifest_sha256": "433e8a0ea5150fc063b7ccd02fc91191ba08032640ec2fd5d63cff5ece129512",
    "source_identity_sha256": "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
    "weights_manifest_fingerprint": "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
    "aligned_manifest_sha256": "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402",
}
LOCAL_SAVED_INT8_EXPERT_QUALIFICATION = {
    "relative_path": "install/local-models/Flash-Next-int8-experts-top64-v1",
    "manifest_sha256": "12593570ee67b62ddeadb951238879b780368cf99e096aa38a7d7771b9dc5c29",
    "plan_sha256": "d0b1f58c87eb4292ea6e7d04eec55f0c722ee7583468dd58e45c2fa3f476d02d",
    "selected_experts_per_layer": 64,
    "minimum_physical_ram_bytes": 256 * 1024**3,
}
_LOCAL_HARDWARE_NOT_PROBED = object()


class LauncherError(RuntimeError):
    pass


class LocalSourceMismatch(LauncherError):
    pass


def _request_json(path, timeout=2):
    request = urllib.request.Request(BASE_URL + path)
    if key := os.environ.get("SPLASH_API_KEY"):
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise LauncherError(
                "Splash authentication failed; set SPLASH_API_KEY to the server's key"
            ) from None
        return None
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        return None


def _running_status():
    status = _request_json("/status", timeout=10)
    if not isinstance(status, dict):
        return None
    return status


def _ensure_installed(model_id):
    if not paths.PACKAGED:
        for command in (
            ["make", "platform-check", "install-environment"],
            ["make", "-j4", "all"],
        ):
            if subprocess.run(command, cwd=ROOT).returncode:
                raise LauncherError("source build failed; see the output above")
    command = [
        str(paths.PYTHON),
        str(ROOT / "install/models.py"),
        "--models",
        str(paths.MODELS),
        "--model",
        model_id,
        "prepare",
    ]
    if subprocess.run(command, cwd=ROOT).returncode:
        raise LauncherError("model download or verification failed")


def _local_file(root, name):
    pure = PurePosixPath(name) if isinstance(name, str) else None
    if (
        pure is None
        or pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or pure.as_posix() != name
    ):
        raise LauncherError("local package contains an invalid file path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise LauncherError("local package file escapes its directory")
    return path


def local_bundle_manifest(package, source=None, *, full_source=False):
    """Check launch metadata; native loading validates the tensor layout."""
    package = Path(package).resolve()
    try:
        manifest = model_artifacts.read_json(package / "manifest.json")
        expected = (package / "manifest.sha256").read_text().strip()
        if (
            manifest.get("schema") != LOCAL_SCHEMA
            or manifest.get("alignment") != model_artifacts.ALIGNMENT
            or not model_artifacts.is_hex_digest(
                manifest.get("source_identity_sha256"), 64
            )
            or expected != model_artifacts.sha256(package / "manifest.json")
        ):
            raise LauncherError("local package has invalid identity or format metadata")
        records = manifest.get("small_files")
        if not isinstance(records, list):
            raise LauncherError("local package has no tokenizer/config metadata")
        required = {
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise LauncherError("local package has invalid metadata records")
            path = _local_file(package, record.get("path"))
            size, digest = record.get("bytes"), record.get("sha256")
            if (
                record["path"] in seen
                or type(size) is not int
                or size <= 0
                or not model_artifacts.is_hex_digest(digest, 64)
                or path.stat().st_size != size
                or model_artifacts.sha256(path) != digest
            ):
                raise LauncherError("local package tokenizer/config metadata changed")
            seen.add(record["path"])
            if source is not None:
                original = _local_file(Path(source).resolve(), record["path"])
                if (
                    original.stat().st_size != size
                    or model_artifacts.sha256(original) != digest
                ):
                    raise LauncherError(
                        "local package does not match the original model metadata"
                    )
        if not required <= seen:
            raise LauncherError("local package is missing tokenizer/config metadata")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise LauncherError("local package has no weight shards")
        if source is not None and full_source:
            print(
                f"Verifying original local-model weights · {len(shards)} shards",
                flush=True,
            )
        for index, record in enumerate(shards, 1):
            if not isinstance(record, dict):
                raise LauncherError("local package has invalid shard metadata")
            path = _local_file(package, record.get("path"))
            size = record.get("bytes")
            if (
                type(size) is not int
                or size <= 0
                or size % model_artifacts.ALIGNMENT
                or path.stat().st_size != size
                or not model_artifacts.is_hex_digest(record.get("sha256"), 64)
                or not model_artifacts.is_hex_digest(record.get("source_sha256"), 64)
            ):
                raise LauncherError("local package weight shard size changed")
            if source is not None:
                original = _local_file(
                    Path(source).resolve(), record.get("source_path")
                )
                source_size = record.get("source_bytes")
                if (
                    type(source_size) is not int
                    or original.stat().st_size != source_size
                ):
                    raise LauncherError(
                        "local package does not match the original model shards"
                    )
                if full_source:
                    print(
                        f"Verifying source shard {index}/{len(shards)} · {original.name}",
                        flush=True,
                    )
                    before = original.stat()
                    digest = model_artifacts.sha256(original)
                    after = original.stat()
                    if (
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ) != (
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ) or digest != record["source_sha256"]:
                        raise LocalSourceMismatch(
                            "local package does not match the original model weight checksums"
                        )
        source_records = [
            {
                "path": record["source_path"],
                "bytes": record["source_bytes"],
                "sha256": record["source_sha256"],
            }
            for record in shards
        ] + [
            {key: record[key] for key in ("path", "bytes", "sha256")}
            for record in records
        ]
        identity = hashlib.sha256(
            json.dumps(
                {
                    "schema": LOCAL_SCHEMA,
                    "source_files": sorted(
                        source_records, key=lambda item: item["path"]
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        if identity != manifest["source_identity_sha256"]:
            raise LauncherError("local package source identity is inconsistent")
    except (
        model_artifacts.ModelError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise LauncherError(
            f"unable to validate local package {package}: {error}"
        ) from error
    return manifest


def _ensure_local_installed(source, package):
    if source is None and package is None:
        raise LauncherError("select --local-model or --local-package")
    source = None if source is None else source.resolve()
    if source is not None and not source.is_dir():
        raise LauncherError("--local-model must name an existing MLX model directory")
    if package is not None:
        package = package.resolve()
        local_bundle_manifest(package, source, full_source=source is not None)
        return package
    packages = (
        paths.DATA / "local-models" if paths.PACKAGED else ROOT / "install/local-models"
    )
    matches = []
    for manifest in sorted(packages.glob("*/manifest.json")):
        try:
            local_bundle_manifest(manifest.parent, source)
        except LauncherError:
            continue
        matches.append(manifest.parent.resolve())
    if len(matches) == 1:
        try:
            local_bundle_manifest(matches[0], source, full_source=True)
        except LocalSourceMismatch:
            pass  # Import a new snapshot while preserving the previous bundle.
        else:
            return matches[0]
    if len(matches) > 1:
        raise LauncherError(
            "multiple local packages match; select one with --local-package"
        )
    importer = ROOT / "dev/tools/import_flash_next.py"
    if not importer.is_file():
        raise LauncherError(
            "no matching local package; supply an imported --local-package"
        )
    identity = model_artifacts.sha256(source / "config.json")[:12]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", source.name).strip("._-") or "local-model"
    alias = f"{name[:108]}-{identity}"
    if (packages / alias).exists() or (packages / alias).is_symlink():
        alias += "-" + secrets.token_hex(3)
    print(
        f"Importing local model into {packages / alias}; original files are preserved.",
        flush=True,
    )
    command = [
        str(paths.PYTHON),
        str(importer),
        str(source),
        "--output-root",
        str(packages),
        "--alias",
        alias,
    ]
    if subprocess.run(command, cwd=ROOT, check=False).returncode:
        raise LauncherError("local model import or verification failed")
    package = packages / alias
    local_bundle_manifest(package, source, full_source=True)
    return package.resolve()


def _ensure_local_runtime(binary):
    if not paths.PYTHON.is_file() and not paths.PACKAGED:
        if subprocess.run(
            ["make", "platform-check", "install-environment"], cwd=ROOT, check=False
        ).returncode:
            raise LauncherError("local serving environment setup failed")
    if (
        not binary.is_file() or not binary.with_name("splash.metallib").is_file()
    ) and not paths.PACKAGED:
        if subprocess.run(
            ["make", "-j4", "flash-next", "BUILD=build/flash-next"],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise LauncherError("native Flash-Next build failed; see the output above")
    if (
        not paths.PYTHON.is_file()
        or not binary.is_file()
        or not binary.with_name("splash.metallib").is_file()
    ):
        raise LauncherError(
            "local serving requires the bundled Python, splash-flash and splash.metallib"
        )


def _local_hardware_identity():
    """CPU-only hardware probe; tests substitute the returned brand/RAM pair."""
    values = []
    for key in ("machdep.cpu.brand_string", "hw.memsize"):
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode:
            return None
        values.append(result.stdout.strip())
    return values[0], int(values[1])


def _apply_local_profile_defaults(environment, defaults):
    """Merge qualified defaults into a copied exec environment.

    Only implied dependent defaults change when a user disables their parent.
    Explicit values, the caller's defaults, and the process environment stay
    intact. Batch prefill is independent of the decode batching switch.
    """
    adjusted = dict(defaults)
    dependencies = {
        "SPLASH_FLASH_QSA_MPP": ("SPLASH_FLASH_QSA_F32",),
        "SPLASH_FLASH_QSA_ROW_TILES": ("SPLASH_FLASH_QSA_MPP", "SPLASH_FLASH_QSA_F32"),
        "SPLASH_FLASH_MTP_QSA_MPP": ("SPLASH_FLASH_MTP_QSA_F32",),
        "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE": (
            "SPLASH_FLASH_FLOAT_DENSE_CACHE",
            "SPLASH_FLASH_QMV_F32",
        ),
        "SPLASH_FLASH_MOE_M64": ("SPLASH_FLASH_MOE_Q4X8",),
        "SPLASH_FLASH_MOE_DIRECT_A": (
            "SPLASH_FLASH_MOE_Q4X8",
            "SPLASH_FLASH_BLOCKED_MOE",
        ),
        "SPLASH_FLASH_SHARED_EXPERT_FUSED": ("SPLASH_FLASH_DENSE_CACHE",),
        "SPLASH_FLASH_DENSE_M64_OUT": ("SPLASH_FLASH_DENSE_CACHE",),
        "SPLASH_FLASH_GDN_BATCH_ILP": (
            "SPLASH_FLASH_GDN_STAGED",
            "SPLASH_FLASH_BATCH_PREFILL",
        ),
        "SPLASH_FLASH_MTP_QMV_F32": ("SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_MTP"),
        "SPLASH_FLASH_HC_UP_F32_MPP": (
            "SPLASH_FLASH_FUSE_HC",
            "SPLASH_FLASH_FLOAT_DENSE_CACHE",
        ),
        "SPLASH_FLASH_GDN_LAZY_ROLLBACK": ("SPLASH_FLASH_FUSE_GDN",),
        # Joint vocabulary follows the batch-MTP implied default. Include its
        # explicit parents here because this merger does not recurse through
        # adjusted defaults. BATCH is launcher policy, not a kernel constraint.
        "SPLASH_FLASH_MTP_Q8_BF16_REGISTER": (
            "SPLASH_FLASH_DENSE_CACHE",
            "SPLASH_FLASH_BATCH_MTP",
            "SPLASH_FLASH_MTP",
            "SPLASH_FLASH_BATCH",
        ),
        # The idle route requires the qualified complete immutable owner union.
        # Exact store "0" values are selectors, not paths. Other caller values
        # stay visible to the native loader and its owner-union validation.
        "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": (
            "SPLASH_FLASH_DENSE_CACHE",
            "SPLASH_FLASH_FLOAT_DENSE_CACHE",
            "SPLASH_FLASH_BLOCKED_MOE",
            "SPLASH_FLASH_OPERAND_STORE",
            "SPLASH_FLASH_INT8_EXPERT_STORE",
        ),
        "SPLASH_FLASH_BATCH_MTP": ("SPLASH_FLASH_MTP", "SPLASH_FLASH_BATCH"),
        "SPLASH_FLASH_BATCH_MTP_PREFILL": (
            "SPLASH_FLASH_MTP",
            "SPLASH_FLASH_BATCH_PREFILL",
        ),
    }
    for child, parents in dependencies.items():
        if child in adjusted and any(
            environment.get(parent) == "0" for parent in parents
        ):
            adjusted[child] = "0"
    # Path-valued implied defaults are removed when disabled, rather than
    # assigning "0" as a filesystem path. An explicit caller path survives.
    if environment.get("SPLASH_FLASH_BLOCKED_MOE") == "0":
        adjusted.pop("SPLASH_FLASH_INT8_EXPERT_STORE", None)
    for key, value in adjusted.items():
        environment.setdefault(key, value)


def _saved_operand_metadata(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise LauncherError(f"saved operand metadata is missing or invalid: {path}")
    return path.read_bytes()


def _qualified_saved_operand_layout(package):
    """Shared CPU witness for the already qualified source/layout/norm triple."""
    qualification = LOCAL_SAVED_OPERAND_QUALIFICATION
    aligned_bytes = _saved_operand_metadata(Path(package) / "manifest.json")
    aligned_digest = hashlib.sha256(aligned_bytes).hexdigest()
    aligned = json.loads(aligned_bytes)
    if (
        not isinstance(aligned, dict)
        or aligned_digest != qualification["aligned_manifest_sha256"]
        or aligned.get("schema") != LOCAL_SCHEMA
        or aligned.get("source_identity_sha256") != qualification["source_identity_sha256"]
        or _saved_operand_metadata(Path(package) / "manifest.sha256") != (aligned_digest + "\n").encode()
    ):
        raise LauncherError("the saved operand default does not match the aligned model layout")
    # The native norm audit qualified this checkpoint as OnePlusWeight.
    # Its effective identity binds source, exact layout and that convention.
    identity = (
        "splash.native-flash-weights-v1\nsource="
        + qualification["source_identity_sha256"]
        + "\nmanifest=" + aligned_digest + "\nnorm=one-plus-weight\n"
    )
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    if fingerprint != qualification["weights_manifest_fingerprint"]:
        raise LauncherError("the saved operand default has an unqualified model fingerprint")
    return fingerprint


def _qualified_saved_operand_defaults(package, environment=None):
    """CPU metadata gate for the one qualified optional saved dense artifact.

    This prepares a qualified optional default without reading payload data.
    Existing caller values are authoritative, including empty invalid values
    which must remain visible to the native loader. Payload verification stays
    in that loader; this helper never reads multi-GB operand payloads.
    """
    environment = os.environ if environment is None else environment
    if "SPLASH_FLASH_OPERAND_STORE" in environment:
        return {}
    qualification = LOCAL_SAVED_OPERAND_QUALIFICATION
    store = ROOT / qualification["relative_path"]
    if not store.exists() and not store.is_symlink():
        return {}

    try:
        if store.is_symlink() or not store.is_dir():
            raise LauncherError("the saved operand default must be a regular directory")
        fingerprint = _qualified_saved_operand_layout(package)
        saved_bytes = _saved_operand_metadata(store / "manifest.json")
        saved_digest = hashlib.sha256(saved_bytes).hexdigest()
        saved = json.loads(saved_bytes)
        if (
            not isinstance(saved, dict)
            or saved_digest != qualification["manifest_sha256"]
            or saved.get("schema") != "splash-local-affine-operands-v1"
            or saved.get("source_identity_sha256") != qualification["source_identity_sha256"]
            or saved.get("weights_manifest_fingerprint") != fingerprint
            or _saved_operand_metadata(store / "manifest.sha256") != (saved_digest + "\n").encode()
        ):
            raise LauncherError("the saved operand default is corrupt or was not qualified for this model")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise LauncherError(f"unable to validate the saved operand default: {error}") from error
    return {"SPLASH_FLASH_OPERAND_STORE": str(store.resolve())}


def _qualified_saved_int8_expert_defaults(
    package, environment=None, *, hardware=_LOCAL_HARDWARE_NOT_PROBED
):
    """Prepare one optional expert store; never activate it or scan payloads.

    Passing the qualified launch hardware tuple reuses the caller's probe.
    The normal qualified profile path passes its existing hardware tuple.
    """
    environment = os.environ if environment is None else environment
    if "SPLASH_FLASH_INT8_EXPERT_STORE" in environment:
        return {}
    if environment.get(
        "SPLASH_FLASH_BLOCKED_MOE", LOCAL_PROFILE["environment"].get("SPLASH_FLASH_BLOCKED_MOE")
    ) != "1":
        return {}
    qualification = LOCAL_SAVED_INT8_EXPERT_QUALIFICATION
    store = ROOT / qualification["relative_path"]
    if not store.exists() and not store.is_symlink():
        return {}
    if hardware is _LOCAL_HARDWARE_NOT_PROBED:
        try:
            hardware = _local_hardware_identity()
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {}
    if (
        not isinstance(hardware, tuple)
        or len(hardware) != 2
        or hardware[0] != "Apple M5 Ultra"
        or type(hardware[1]) is not int
        or hardware[1] < qualification["minimum_physical_ram_bytes"]
    ):
        return {}
    try:
        if store.is_symlink() or not store.is_dir():
            raise LauncherError("the saved INT8 expert default must be a regular directory")
        _qualified_saved_operand_layout(package)
        saved_bytes = _saved_operand_metadata(store / "manifest.json")
        saved = json.loads(saved_bytes)
        model = LOCAL_SAVED_OPERAND_QUALIFICATION
        if (
            not isinstance(saved, dict)
            or hashlib.sha256(saved_bytes).hexdigest() != qualification["manifest_sha256"]
            or saved.get("schema") != "splash-flash-int8-expert-store-v1"
            or saved.get("source_identity_sha256") != model["source_identity_sha256"]
            or saved.get("source_manifest_sha256") != model["aligned_manifest_sha256"]
            or saved.get("plan_sha256") != qualification["plan_sha256"]
            or type(saved.get("alignment")) is not int
            or saved["alignment"] != 16384
            or type(saved.get("target_layers")) is not int
            or saved["target_layers"] != 48
        ):
            raise LauncherError("the saved INT8 expert default has unqualified source, plan, shape or checksum")
        selected = saved.get("selected_experts")
        layers = saved.get("layers")
        selected_count = qualification["selected_experts_per_layer"]
        if not isinstance(selected, list) or not isinstance(layers, list) or len(selected) != 48 or len(layers) != 48:
            raise LauncherError("the saved INT8 expert default must contain 48 selected layer views")
        for index, (ids, layer) in enumerate(zip(selected, layers)):
            if (
                not isinstance(ids, list)
                or len(ids) != selected_count
                or any(type(value) is not int or not 0 <= value < 512 for value in ids)
                or ids != sorted(set(ids))
                or not isinstance(layer, dict)
                or type(layer.get("layer_index")) is not int
                or layer["layer_index"] != index
                or not isinstance(layer.get("projections"), dict)
                or set(layer["projections"]) != {"gate_proj", "up_proj", "down_proj"}
            ):
                raise LauncherError("the saved INT8 expert default has invalid selected experts or layer views")
            for role, (outputs, inputs) in {
                "gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)
            }.items():
                projection = layer["projections"][role]
                shapes = {
                    "dimensions": [selected_count, outputs, inputs],
                    "codes": [selected_count, outputs, inputs],
                    "scales": [selected_count, outputs],
                }
                if not isinstance(projection, dict) or projection.get("source_prefix") != (
                    f"language_model.model.layers.{index}.mlp.switch_mlp.{role}"
                ):
                    raise LauncherError("the saved INT8 expert default has a mismatched source projection")
                for field, expected in shapes.items():
                    value = projection.get(field)
                    if field != "dimensions":
                        if not isinstance(value, dict) or value.get("dtype") != (
                            "I8" if field == "codes" else "F32"
                        ):
                            raise LauncherError("the saved INT8 expert default has an unqualified operand dtype")
                        value = value.get("shape")
                    if not isinstance(value, list) or any(type(item) is not int for item in value) or value != expected:
                        raise LauncherError("the saved INT8 expert default has an unqualified operand shape")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise LauncherError(f"unable to validate the saved INT8 expert default: {error}") from error
    return {"SPLASH_FLASH_INT8_EXPERT_STORE": str(store.resolve())}


def _local_profile_v5_candidate(package, environment=None):
    """Return a historical v5 review copy; does not write a profile/environment."""
    profile = {
        **LOCAL_PROFILE,
        "profile": "m5-ultra-flash-next-v5",
        "minimum_physical_ram_bytes": 192 * 1024**3,
        "environment": {
            **LOCAL_PROFILE["environment"],
            "SPLASH_FLASH_MOE_DIRECT_A": "1",
            "SPLASH_FLASH_QSA_ROW_TILES": "1",
            "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT": "1",
        },
    }
    for key in (
        "SPLASH_FLASH_SHARED_EXPERT_FUSED",
        "SPLASH_FLASH_DENSE_M64_OUT",
        "SPLASH_FLASH_GDN_BATCH_ILP",
        "SPLASH_FLASH_MTP_QMV_F32",
        "SPLASH_FLASH_HC_UP_F32_MPP",
        "SPLASH_FLASH_GDN_LAZY_ROLLBACK",
        "SPLASH_FLASH_MTP_Q8_BF16_REGISTER",
        "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE",
        "SPLASH_FLASH_PLE_SSD_STREAMING",
    ):
        profile["environment"].pop(key, None)
    profile["environment"].update(_qualified_saved_operand_defaults(package, environment))
    return profile


def _local_profile_v6_candidate(package, environment=None):
    """Return a historical v6 review copy without writing a profile/environment."""
    profile = _local_profile_v5_candidate(package, environment)
    profile["profile"] = "m5-ultra-flash-next-v6"
    profile["environment"].update({
        "SPLASH_FLASH_SHARED_EXPERT_FUSED": "1",
        "SPLASH_FLASH_DENSE_M64_OUT": "1",
        "SPLASH_FLASH_GDN_BATCH_ILP": "1",
        "SPLASH_FLASH_MTP_QMV_F32": "1",
    })
    return profile


def _local_profile_v7_candidate(package, environment=None):
    """Return a historical v7 review copy without writing a profile/environment."""
    profile = _local_profile_v6_candidate(package, environment)
    profile["profile"] = "m5-ultra-flash-next-v7"
    profile["environment"].update({
        "SPLASH_FLASH_HC_UP_F32_MPP": "1",
        "SPLASH_FLASH_GDN_LAZY_ROLLBACK": "1",
    })
    return profile


def _local_profile_v8_candidate(package, environment=None):
    """Return a private v8 review copy; does not activate defaults or write files."""
    profile = _local_profile_v7_candidate(package, environment)
    profile["profile"] = "m5-ultra-flash-next-v8"
    profile["environment"]["SPLASH_FLASH_MTP_Q8_BF16_REGISTER"] = "1"
    return profile


def _local_profile_v9_candidate(package, environment=None):
    """Return a v9 review copy without writing the accepted serving profile.

    Native startup validates the exact immutable source and owner union. The
    256 GiB proposal gate qualifies this machine's memory headroom; historical
    review copies and the currently accepted profile retain their own gates.
    Optional derived paths are still selected by the accepted launch helper,
    not forced into this static proposal.
    """
    profile = _local_profile_v8_candidate(package, environment)
    profile["profile"] = "m5-ultra-flash-next-v9"
    profile["minimum_physical_ram_bytes"] = 256 * 1024**3
    profile["environment"]["SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"] = "1"
    return profile


def _local_profile_v10_candidate(package, environment=None):
    """Return the SSD-placement review copy without activating defaults."""
    profile = _local_profile_v9_candidate(package, environment)
    profile["profile"] = "m5-ultra-flash-next-v10"
    profile["environment"]["SPLASH_FLASH_PLE_SSD_STREAMING"] = "1"
    return profile


def _local_profile_defaults(package):
    """Apply only the known, measured local profile; no remote model roster."""
    try:
        profile = model_artifacts.read_json(ROOT / ".splash-local-profile.json")
        if (
            type(profile.get("schema_version")) is not int
            or type(profile.get("minimum_physical_ram_bytes")) is not int
            or profile != LOCAL_PROFILE
        ):
            return {}
        manifest = local_bundle_manifest(package)
        config = model_artifacts.read_json(package / "config.json")
        if (
            manifest.get("schema") != LOCAL_SCHEMA
            or manifest["source_identity_sha256"] != profile["source_identity_sha256"]
            or config.get("model_type") != profile["architecture"]
        ):
            return {}
        hardware = _local_hardware_identity()
        if (
            hardware is None
            or hardware[0] != profile["cpu_brand"]
            or type(hardware[1]) is not int
            or hardware[1] < profile["minimum_physical_ram_bytes"]
        ):
            return {}
    except (
        model_artifacts.ModelError,
        LauncherError,
        OSError,
        KeyError,
        ValueError,
        subprocess.TimeoutExpired,
    ):
        return {}
    defaults = dict(profile["environment"])
    # Profile gate probes hardware once. Optional path-valued defaults retain
    # explicit caller choices and use the same source/layout witness; malformed
    # present artifacts propagate an error rather than silently losing defaults.
    defaults.update(_qualified_saved_operand_defaults(package, os.environ))
    defaults.update(_qualified_saved_int8_expert_defaults(package, os.environ, hardware=hardware))
    return defaults


def _serve_lock_owner(lock):
    try:
        lock.seek(0)
        owner = json.load(lock)
    except (OSError, UnicodeError, ValueError):
        return ""
    if not isinstance(owner, dict):
        return ""
    pid, model, port = owner.get("pid"), owner.get("model"), owner.get("port")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(model, str)
        or not model
        or not model.isprintable()
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        return ""
    return f" (PID {pid}, model {model}, port {port})"


def serve(args):
    # Keep this descriptor across exec: the foreground server owns the lock
    # until it exits.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    port = args.port
    lock_name = "serve.lock" if port == PORT else f"serve-{port}.lock"
    with (RUNTIME_DIR / lock_name).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError(
                f"Splash is already serving{_serve_lock_owner(lock)}; "
                "stop it with Ctrl+C first"
            ) from None
        lock.seek(0)
        lock.truncate()
        json.dump({"pid": os.getpid(), "model": args.model, "port": port}, lock)
        lock.flush()
        # Fail before downloads/builds if another service owns the default port.
        # The HTTP server also binds before loading weights, closing the race.
        with socket.socket() as probe:
            # Match the HTTP listener: closed connections in TIME_WAIT must
            # not block a restart; a live listener still owns the address.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                raise LauncherError(
                    f"127.0.0.1:{port} is in use; stop that service first"
                ) from None
        local = args.local_model is not None or args.local_package is not None
        if local:
            binary = ROOT / (
                "engine/splash-flash"
                if paths.PACKAGED
                else "build/flash-next/splash-flash"
            )
            _ensure_local_runtime(binary)
            root = _ensure_local_installed(args.local_model, args.local_package)
            model_arguments = ["--local-package", str(root), "--tokenizer", str(root)]
        else:
            _ensure_installed(args.model)
            root = model_artifacts.installed_root(paths.MODELS, args.model)
            binary = paths.BINARY
            model_arguments = [
                str(root / "target"),
                str(root / "draft"),
                "--tokenizer",
                str(root / "tokenizer"),
            ]
        command = [
            str(paths.PYTHON),
            "-u",
            str(ROOT / "server/server.py"),
            *model_arguments,
            "--model",
            args.model,
            "--binary",
            str(binary),
            "--port",
            str(port),
            "--max-memory",
            "auto" if args.max_memory is None else str(args.max_memory),
            "--max-context",
            "auto" if args.max_context is None else str(args.max_context),
        ]
        if args.max_image_pixels is not None:
            command.extend(["--max-image-pixels", str(args.max_image_pixels)])
        if args.no_webui:
            command.append("--no-webui")
        for host in args.allowed_host:
            command.extend(["--allowed-host", host])
        environment = dict(
            os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error"
        )
        if local:
            defaults = _local_profile_defaults(root)
            _apply_local_profile_defaults(environment, defaults)
            # Explicit placement choices override the qualified local profile.
            if args.ple_ssd_streaming is not None:
                environment["SPLASH_FLASH_PLE_SSD_STREAMING"] = (
                    "1" if args.ple_ssd_streaming else "0"
                )
            if args.ple_ssd_cache_mb is not None:
                if environment.get("SPLASH_FLASH_PLE_SSD_STREAMING") != "1":
                    raise LauncherError(
                        "--ple-ssd-cache-mb requires SSD streaming to be enabled "
                        "by the qualified local profile, --ple-ssd-streaming, "
                        "or SPLASH_FLASH_PLE_SSD_STREAMING=1"
                    )
                environment["SPLASH_FLASH_PLE_SSD_CACHE_MB"] = str(args.ple_ssd_cache_mb)
            if defaults:
                print(f"Local runtime profile · {LOCAL_PROFILE['profile']}", flush=True)
        if args.api_key is not None:
            environment["SPLASH_API_KEY"] = args.api_key
        # Detached, because execve replaces this process a line later and a
        # thread would not survive it. Failure is silent by design.
        if not local:
            catalog.spawn_refresh()
        os.set_inheritable(lock.fileno(), True)
        os.execve(command[0], command, environment)


def coding_client(args):
    path = clients.find_executable(args.command)
    snapshot = _running_status()
    if snapshot is None:
        raise LauncherError(
            "No ready Splash server. Run 'splash serve --model <HF_REPO_ID>' "
            "in another terminal first."
        )
    catalog = _request_json("/v1/models")
    models = catalog.get("data", []) if isinstance(catalog, dict) else []
    if (
        not isinstance(models, list)
        or len(models) != 1
        or not isinstance(models[0], dict)
        or models[0].get("owned_by") != "splash"
    ):
        raise LauncherError("Could not identify the local Splash server")
    model, context = models[0].get("id"), snapshot.get("maximum_context_tokens")
    if type(context) is not int or context <= 0:
        raise LauncherError(
            "Splash is running but its context limit is not available yet; wait and retry"
        )
    command, environment = clients.command(
        args.command,
        path,
        BASE_URL,
        model,
        context,
        RUNTIME_DIR,
        client_args=args.client_args,
    )
    print(f"Starting {args.command}: {model} · {context:,} context tokens", flush=True)
    if args.command == "claude":
        print(
            "Claude hosted WebSearch is unavailable. "
            "WebFetch, local tools and MCP are unchanged.",
            flush=True,
        )
    elif args.command == "codex":
        print(
            "Codex hosted WebSearch is disabled: Splash does not provide "
            "OpenAI's search service. Local tools and MCP are unchanged.",
            flush=True,
        )
    os.execvpe(path, command, environment)


def _parse_max_memory(value):
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    suffixes = {
        unit + suffix: 1024**power
        for power, unit in enumerate(("K", "M", "G"), 1)
        for suffix in ("", "B", "IB")
    }
    multiplier = 1
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            normalized, multiplier = normalized[: -len(suffix)], suffixes[suffix]
            break
    try:
        result = int(normalized) * multiplier
    except ValueError:
        raise argparse.ArgumentTypeError("use a value such as 32G") from None
    if not 1 <= result <= 2**63 - 1:
        raise argparse.ArgumentTypeError("use a positive value such as 32G")
    return result


def _parse_max_context(value):
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    try:
        result = (
            int(normalized[:-1]) * 1024 if normalized.endswith("K") else int(normalized)
        )
    except ValueError:
        raise argparse.ArgumentTypeError("use a value such as 100K") from None
    if not 1 <= result <= 262144:
        raise argparse.ArgumentTypeError("must be between 1 and 256K tokens")
    return result


def _version():
    if not paths.PACKAGED:
        return "Splash (source checkout)"
    return "Splash " + str(
        json.loads((paths.ROOT / "release.json").read_text())["version"]
    )


def _parse_ple_ssd_cache_mb(value):
    if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise argparse.ArgumentTypeError("PLE SSD cache must be a decimal MiB size from 0 to 1024")
    size = int(value)
    if not 0 <= size <= 1024:
        raise argparse.ArgumentTypeError("PLE SSD cache must be a decimal MiB size from 0 to 1024")
    return size


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    client_args = []
    if argv and argv[0] in clients.INSTALL_URLS:
        argv, client_args = argv[:1], argv[1:]
        if client_args[:1] == ["--"]:
            client_args = client_args[1:]
    elif "--" in argv:
        boundary = argv.index("--")
        argv, client_args = argv[:boundary], argv[boundary + 1 :]
    parser = argparse.ArgumentParser(prog="splash", description=__doc__)
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="run the local server; Ctrl+C stops it")
    server.add_argument(
        "--model",
        type=model_artifacts.parse_repo_id,
        required=True,
        metavar="OWNER/REPO",
        help="served owner/repo ID (a local alias when using --local-model)",
    )
    server.add_argument(
        "--local-model", type=Path, help="existing local Flash-Next MLX directory"
    )
    server.add_argument(
        "--local-package", type=Path, help="imported local Flash-Next bundle"
    )
    placement = server.add_mutually_exclusive_group()
    placement.add_argument(
        "--ple-ssd-streaming", action="store_true", default=None,
        help="read n-gram table rows from SSD (default for the qualified local Flash-Next profile)",
    )
    placement.add_argument(
        "--no-ple-ssd-streaming", action="store_false", dest="ple_ssd_streaming",
        help="keep the n-gram table in GPU-accessible memory (local Flash-Next only)",
    )
    server.add_argument(
        "--ple-ssd-cache-mb", type=_parse_ple_ssd_cache_mb, metavar="MIB",
        help="bounded n-gram SSD row cache in MiB, 0 to 1024 (default: 64)",
    )
    server.add_argument(
        "--port", type=int, default=PORT, help="local HTTP port (default: 8000)"
    )
    server.add_argument(
        "--max-memory",
        type=_parse_max_memory,
        help="Metal budget ceiling, e.g. 28G (default: auto)",
    )
    server.add_argument(
        "--max-context",
        type=_parse_max_context,
        help="context limit, e.g. 100K (default: auto)",
    )
    server.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="additional HTTP Host name to accept (repeatable)",
    )
    server.add_argument(
        "--max-image-pixels", type=int, help="maximum resized pixels per image"
    )
    server.add_argument(
        "--api-key",
        default=os.environ.get("SPLASH_API_KEY"),
        help="API key (default: SPLASH_API_KEY environment variable)",
    )
    server.add_argument("--no-webui", action="store_true", help="disable the chat page")
    for name in clients.INSTALL_URLS:
        commands.add_parser(name, help=f"connect {name} to the running server")
    args = parser.parse_args(argv)
    if args.command == "serve" and not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.command == "serve":
        local = args.local_model is not None or args.local_package is not None
        if (args.ple_ssd_streaming is not None or args.ple_ssd_cache_mb is not None) and not local:
            parser.error("PLE SSD options require --local-model or --local-package")
    if args.command == "serve" and args.api_key is not None:
        if not args.api_key or any(ord(c) <= 32 or ord(c) >= 127 for c in args.api_key):
            parser.error("API key must contain only visible ASCII characters")
    if client_args and args.command == "serve":
        parser.error("arguments after -- are only supported for coding clients")
    args.client_args = client_args
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return serve(args) if args.command == "serve" else coding_client(args)
    except (LauncherError, clients.ClientError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
