#!/usr/bin/env python3
"""Entrypoint for the frozen HC/compact adapter; original CLI stays intact."""
import argparse
import importlib.util
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--build', required=True, type=Path)
args, rest = parser.parse_known_args()
build = args.build.resolve()
path = build / 'source/dev/benchmarks/hc_pad_verify_worker_sep22/semantic_quality.py'
spec = importlib.util.spec_from_file_location('_frozen_combined_hc_semantic', path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
runner = adapter.load(build)
if rest and rest[0] == 'measure' and not any(token == '--runtime-build' or token.startswith('--runtime-build=') for token in rest):
    rest += ['--runtime-build', str(build)]
raise SystemExit(runner.main(rest))
