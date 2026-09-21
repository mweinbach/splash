"""Prepare CPU-only source geometry plans; generated GPU commands are root-run."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(package) or package.is_relative_to(output):
        raise ValueError("screen artifacts must be outside the model package")
    manifest_bytes = (package / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    tensors, quantization = manifest["tensors"], manifest["quantization"]
    projections = []

    def add(prefix: str, role: str) -> None:
        if prefix + ".weight" not in tensors:
            return
        weight, scales = tensors[prefix + ".weight"], tensors[prefix + ".scales"]
        quant = quantization.get(prefix, quantization)
        rows = weight["shape"][0]
        inputs = scales["shape"][1] * quant["group_size"]
        if weight["dtype"] != "U32" or rows % 64 or inputs % 32 or inputs > 32768:
            raise ValueError(f"ineligible float small-row geometry: {prefix}")
        projections.append({"prefix": prefix, "role": role, "n": rows, "k": inputs,
                            "bits": quant["bits"], "group_size": quant["group_size"]})

    for layer in range(48):
        base = f"language_model.model.layers.{layer}"
        for kind in ("attn_hyper_connection", "mlp_hyper_connection"):
            for direction in ("down", "up"):
                role = f"input_mix_weight_{direction}"
                add(f"{base}.{kind}.{role}", role)
        for kind in ("in_proj_qkv", "in_proj_z", "out_proj"):
            add(f"{base}.linear_attn.{kind}", f"linear_attn.{kind}")
        for kind in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"):
            add(f"{base}.self_attn.{kind}", f"self_attn.{kind}")
        for kind in ("gate_proj", "up_proj", "down_proj"):
            add(f"{base}.mlp.shared_expert.{kind}", f"mlp.shared_expert.{kind}")
        for kind in ("key_proj", "value_proj"):
            add(f"{base}.ple.{kind}", f"ple.{kind}")
    for direction in ("down", "up"):
        role = f"input_mix_weight_{direction}"
        add(f"language_model.model.hyper_connection_mixer.{role}", f"final_hc.{role}")

    unique = {}
    for projection in projections:
        key = tuple(projection[x] for x in ("role", "n", "k", "bits", "group_size"))
        unique.setdefault(key, projection)
    expanded = list(unique.values())
    quick = [p for p in expanded if any(f".layers.{layer}." in p["prefix"]
                                      for layer in (0, 1, 3)) or p["role"].startswith("final_hc.")]
    library_bytes = args.library.read_bytes()
    required_controls = ["flash_affine_mlx_qmv_f32xsum_v1_" + name
                         for name in ("q4_g64", "q5_g128", "q5_g64", "q6_g64")]
    if any(name.encode() not in library_bytes for name in required_controls):
        raise ValueError("library lacks current dense QMV control name strings")
    flags = {"SPLASH_FLASH_QMV_F32": "1", "FLASH_FLOAT_CACHE_ROWS": "4,8,16",
             "FLASH_FLOAT_CACHE_TILES": "0,2", "FLASH_FLOAT_CACHE_REPEATS": "6",
             "FLASH_FLOAT_CACHE_CPU_COLUMNS": "19", "FLASH_FLOAT_CACHE_BODY_BOUNDARY_AUDIT": "1",
             "FLASH_FLOAT_CACHE_CONTINUE_ON_ACCURACY_FAILURE": "1"}
    output.mkdir(parents=True, exist_ok=True)
    commands = {}
    for mode, selected in (("quick", quick), ("expanded", expanded)):
        assignments = flags | {"FLASH_FLOAT_CACHE_PREFIXES": ",".join(p["prefix"] for p in selected)}
        command = shlex.join(["env", *(f"{key}={value}" for key, value in assignments.items()),
                              str(args.oracle.resolve()), str(args.library.resolve()), str(package),
                              str(output / f"float-body-{mode}-screen.json")])
        commands[mode] = command
        (output / f"run-float-body-{mode}-screen.sh").write_text(
            "#!/bin/sh\n# Root owns GPU scheduling; this file is never run by the plan preparer.\n"
            "exec " + command + "\n")
    plan = {"source_identity_sha256": manifest["source_identity_sha256"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "library_path": str(args.library.resolve()),
            "library_sha256": hashlib.sha256(library_bytes).hexdigest(),
            "eligible_body_matrices": len(projections), "quick": quick, "expanded": expanded,
            "command_flags": flags, "commands": commands,
            "router": {"dtype": "BF16", "n": 512, "k": 2560,
                       "included": False, "reason": "stored BF16, not an affine projection"},
            "excluded": ["vocabulary head (already separate strict precision proof)",
                         "selected experts (separate cache implementation)",
                         "HC block injection and GDN A/B (N not divisible by 64)"],
            "gpu_commands_executed": 0}
    (output / "float-body-source-screen-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(output / "float-body-source-screen-plan.json"),
                      "quick_source_groups": len(quick), "expanded_source_groups": len(expanded),
                      "eligible_body_matrices": len(projections), "gpu_commands_executed": 0}))


if __name__ == "__main__":
    main()
