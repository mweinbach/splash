#!/usr/bin/env python3
"""CPU-only full current-header restoration build, after exact BB09 baseline."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbols(path):
    output = subprocess.run(["xcrun", "-sdk", "macosx", "metal-nm", "--defined-only", str(path)],
        capture_output=True, text=True, check=True).stdout
    return sorted(line.split()[-1] for line in output.splitlines() if len(line.split()) >= 3 and line.split()[-2] == "T")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "build/batch-prefill-restoration-plan-sep22-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/batch-prefill-restored-teacher-clock-sep22-worker-v1")
    args = parser.parse_args()
    plan_dir, output = args.plan.resolve(), args.output.resolve()
    plan = json.loads((plan_dir / "restoration-plan.json").read_text())
    baseline = json.loads((plan_dir / "original76-candidate-first-baseline-validation.json").read_text())
    lineage = json.loads((plan_dir / "original76AIR-dense-leaf-lineage.json").read_text())
    if output.exists() or not baseline["pass"] or baseline["actual_baseline_metallib_sha256"] != lineage["original_current_metallib_sha256"]:
        raise ValueError("fresh output and exact preregistered original76 BB09 baseline required")
    parent = Path(plan["qualified_math_parent"])
    clock = Path(plan["clock_guard_parent"])
    manifest = json.loads((parent / "overlay-manifest.json").read_text())
    records = []
    for record in manifest["files"]:
        relative = Path(record["path"])
        source = parent / "source" / relative
        if sha(source) != record["sha256"]:
            raise ValueError("current parent source drift: " + str(relative))
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # The parent sparse overrides are inherited BEFORE applying the reviewed
    # restoration changes, so neither clock nor GDN guard is rolled back.
    for relative in ("runtime/flash/FlashWorker.mm", "runtime/flash/FlashGDNBatchILP.cpp",
        "dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp"):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clock / "source" / relative, destination)
    for relative in plan["changed_paths"]:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_dir / "proposed" / relative, destination)
    # Pin private oracle templates and its included helpers in this same tree.
    for relative in ("dev/benchmarks/prefill_batch_teacher_sep21/oracle.mm",
        "dev/benchmarks/prefill4k_attribution.mm"):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for path in sorted((output / "source").rglob("*")):
        if path.is_file(): records.append({"path": str(path.relative_to(output / "source")), "sha256": sha(path)})
    core = []
    for record in manifest["frozen_objects"]:
        source = parent / record["path"]
        if sha(source) != record["sha256"]: raise ValueError("current core object drift")
        destination = output / "core" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination); core.append(destination)
    # Freeze the exact original76, placing AB FIRST as baseline validation did.
    airs = []
    ordered = [lineage["original_AIR_inputs"][-1], *lineage["original_AIR_inputs"][:-1]]
    for index, record in enumerate(ordered):
        source = Path(record["frozen_path"])
        if sha(source) != record["sha256"]: raise ValueError("original current AIR drift")
        destination = output / "original-air" / f"{index:03d}-{source.name}"
        destination.parent.mkdir(exist_ok=True); shutil.copy2(source, destination); airs.append(destination)
    leaf = output / "source/runtime/metal/kernels/shared/flash_dense_cache_prefill.metal"
    narrow = Path(lineage["dense_leaf"]["archived_source"]).read_text()
    if leaf.read_text().replace("(p.rows != 2048 && p.rows != 4096 && p.rows != 8192)", "p.rows != 2048") != narrow:
        raise ValueError("reviewed dense leaf differs beyond wide row guard")
    wide_air = output / "wide-dense-prefill.air"
    metal_command = ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.1", "-O3", "-Wall", "-Wextra", "-Werror",
        f"-I{output / 'source/runtime'}", "-mmacosx-version-min=27.0", "-c", str(leaf), "-o", str(wide_air)]
    subprocess.run(metal_command, cwd=ROOT, check=True)
    original_leaf = next(path for path in airs if "flash_dense_cache_prefill.air" in path.name)
    if symbols(original_leaf) != symbols(wide_air): raise ValueError("dense leaf exported symbols changed")
    actual_airs = [wide_air if path == original_leaf else path for path in airs]
    library = output / "splash.metallib"
    link_metal = ["xcrun", "-sdk", "macosx", "metallib", *map(str, actual_airs), "-o", str(library)]
    subprocess.run(link_metal, cwd=ROOT, check=True)
    old_symbols = symbols(plan_dir / "preregistered-original76-candidate-first-baseline.metallib")
    new_symbols = symbols(library)
    if old_symbols != new_symbols: raise ValueError("current pipeline symbol closure changed")
    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", "-Wno-deprecated-declarations",
        "-fobjc-arc", "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1",
        f"-I{output / 'source'}", f"-I{output / 'source/runtime'}",
        f"-I{output / 'source/dev/benchmarks/prefill4k_attention'}"]
    command = ["xcrun", "-sdk", "macosx", "clang++", *flags]
    rebuild = [*manifest["rebuild"], {"object": "teacher_bulk", "source": "dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp"}]
    if len(rebuild) != 50: raise ValueError("all50 current header consumers required")
    def compile_one(record):
        source = output / "source" / record["source"]
        destination = output / "host" / (record["object"] + ".o")
        destination.parent.mkdir(exist_ok=True)
        invocation = [*command, "-MMD", "-MP", "-c", str(source), "-o", str(destination)]
        subprocess.run(invocation, cwd=ROOT, check=True)
        return {"object": str(destination.relative_to(output)), "sha256": sha(destination),
            "source": record["source"], "source_sha256": sha(source)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        compiled = list(pool.map(compile_one, rebuild))
    objects = [output / record["object"] for record in compiled]
    frameworks = ["-framework", "Foundation", "-framework", "Metal", "-framework", "IOKit"]
    subprocess.run([*command, *map(str, objects), *map(str, core), *frameworks, "-o", str(output / "splash-flash")], cwd=ROOT, check=True)
    nonworker = [path for path in objects if path.name != "FlashWorker.o"]
    for name, relative in (("head-cache-future-oracle", "dev/benchmarks/prefill_batch_teacher_sep21/oracle.mm"),
        ("batch-main-oracle", "dev/benchmarks/flash_batch_prefill_oracle.mm")):
        subprocess.run([*command, str(output / "source" / relative), *map(str, nonworker), *map(str, core),
            *frameworks, "-o", str(output / name)], cwd=ROOT, check=True)
    cpu = subprocess.run([str(output / "splash-flash"), "--cpu-self-test"], capture_output=True, text=True, check=True)
    # Verify every compiler-owned dependency is in the frozen current tree.
    known = {record["path"] for record in records}
    for dep in (output / "host").glob("*.d"):
        import shlex
        tokens = shlex.split(dep.read_text().replace("\\\n", " ").splitlines()[0].split(":", 1)[1])
        for token in tokens:
            path = Path(token).resolve()
            if output / "source" in path.parents and str(path.relative_to(output / "source")) not in known:
                raise ValueError("compiler dependency not frozen: " + str(path))
    seal = {"schema": "splash-batch-prefill-current-header-restoration-cpu-seal-v1", "pass": True,
        "gpu_executed": False, "model_operand_payload_bytes_read": 0,
        "whole_worker_GPU_qualification_complete": False, "parent": str(parent), "clock_guard_parent": str(clock),
        "baseline_exact_BB09_first76_recipe_verified_before_wide_compile": True,
        "source_files": records, "host_TUs_rebuilt": len(compiled), "compiled_objects": compiled,
        "core_objects": [{"path": str(path.relative_to(output)), "sha256": sha(path)} for path in core],
        "effective_objects": len(objects) + len(core), "all_current_header_consumers_recompiled": True,
        "original76_AIR_inputs_retained": len(airs), "only_replaced_AIR": str(original_leaf.relative_to(output)),
        "wide_dense_AIR_sha256": sha(wide_air), "shader_numeric_body_and_ABI_unchanged": True,
        "all_exported_pipeline_symbols_preserved": True, "pipeline_symbol_count": len(new_symbols),
        "preserved_pipeline_symbols": new_symbols, "metallib_sha256": sha(library),
        "binary_sha256": sha(output / "splash-flash"), "metal_compile_command": metal_command,
        "metallib_link_command": link_metal, "compiler_flags": flags, "worker_CPU": json.loads(cpu.stdout),
        "head_oracle_sha256": sha(output / "head-cache-future-oracle"),
        "main_oracle_sha256": sha(output / "batch-main-oracle"),
        "group_teacher_extra_workspace_bytes": 0, "batch_main_extra_workspace_bytes": 234356736,
        "scalar_bulk_planned_workspace_bytes_preserved": 310181888,
        "production_files_modified": False}
    (output / "compiled-cpu-seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps({key: value for key, value in seal.items() if key not in
        ("source_files", "compiled_objects", "preserved_pipeline_symbols", "worker_CPU", "metal_compile_command", "metallib_link_command")}))


if __name__ == "__main__":
    main()
