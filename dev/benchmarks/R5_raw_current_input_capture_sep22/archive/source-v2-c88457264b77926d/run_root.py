#!/usr/bin/env python3
"""Root-only, sequential real capture-off/on request and streamed-byte proof.

Import/source checking never reads model, token, capture or actual report data.
There is no standalone draft generator or tensor recomputation here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path('/Users/mweinbach/Projects/splash')
TOKEN_FILE = ROOT / 'build/release/flash/sep22-fixed4-qualified-http-exact2048-Root-tokens-v1.json'
TOKEN_SHA = '72e5a23f0504ba862d43c22e01e820fc4007b3b939ec4cbc8518aaee499832fb'
TOKEN_U32_SHA = '55cf1a355b4a2c97012c752b87955198ef3bb1f1b992b3fb48d35ff7659f3795'
LIBRARY = 'dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6'
CAPTURE = 'SPLASH_FLASH_CAPTURE_R5_RAW_INPUT_SEP22'
PROOF = 'SPLASH_FLASH_CAPTURE_R5_RAW_PROOF_SEP22'
DIRECTORY = 'SPLASH_FLASH_CAPTURE_R5_RAW_DIRECTORY_SEP22'
FRAMES = ('selected_pending', 'selected_commit', 'next_real_target')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested(value, path):
    for field in path.split('.'):
        value = value[field]
    return value


def group_gone(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class TrackedProcess:
    def __init__(self, process):
        self.process = process
        self.stdin, self.stdout = process.stdin, process.stdout
        self.terminate_called = self.kill_called = False

    def __getattr__(self, name):
        return getattr(self.process, name)

    def terminate(self):
        self.terminate_called = True
        return self.process.terminate()

    def kill(self):
        self.kill_called = True
        return self.process.kill()


def run_arm(command, name, tokens):
    # Runtime and token/capture data are Root-only beyond the explicit main gate.
    sys.path.insert(0, str(ROOT))
    from server.runtime import MultiplexedRuntime, GenerationRequest, Deadline
    from server import protocol as wire
    arm = command['arms'][name]
    directory, log_path = Path(arm['directory']), Path(arm['native_log'])
    require(not directory.exists() and not log_path.exists(), 'fresh arm directory/log required')
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('SPLASH_', 'FLASH_'))}
    environment.update(command['environment'])
    environment[CAPTURE], environment[PROOF], environment[DIRECTORY] = ('0' if name == 'control' else '1'), '1', str(directory)
    processes = []
    runtime = None
    errors = []
    result_info = {}
    with log_path.open('x') as log:
        def create():
            require(not processes, 'automatic native restart forbidden')
            process = subprocess.Popen([str(Path(command['build']) / 'splash-flash'), 'serve-flash-native', command['package'], '16384', 'auto'], cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0, start_new_session=True)
            tracked = TrackedProcess(process)
            processes.append(tracked)
            return tracked

        def idle():
            until = time.monotonic() + 30
            while time.monotonic() < until:
                status = json.loads(runtime.status(timeout=5).json)
                require(status['ready'] is True and status['memory_pressure'] == 'normal', 'actual worker health/pressure invalid')
                queue = status['scheduler']
                if not queue['active_requests'] and not queue['queued'] and not queue['command_in_flight']:
                    return status
                time.sleep(.02)
            raise TimeoutError('worker did not reach safe idle')

        try:
            runtime = MultiplexedRuntime(process_factory=create, startup_timeout=300, io_timeout=5)
            runtime.wait_ready(timeout=300)
            before = idle()
            require(nested(before, 'mtp.singleton_maximum_draft_tokens') == 4, 'actual singleton cap4 required')
            call = runtime.submit(GenerationRequest(tuple(tokens), 64, Deadline.after(300), cohort=wire.Cohort.GREEDY))
            require(call.request_id == 1, 'first and only native request must be1')
            result = call.result(timeout=330)
            after = idle()
            require(result.start is not None and result.start.request_id == 1 and result.start.cache_disposition == wire.CacheDisposition.MISS and result.start.matched_prompt_tokens == 0, 'actual cacheMISS/matched0 required')
            require(result.done.prompt_tokens == 2048 and result.done.completion_tokens == 64 and len(result.tokens) == 64, 'one actual2048/64 request required; no budget growth')
            require(nested(after, 'metrics.prefill_input_tokens') - nested(before, 'metrics.prefill_input_tokens') == 2048 and nested(after, 'metrics.autoregressive_output_tokens') - nested(before, 'metrics.autoregressive_output_tokens') == 64, 'actual native token counters differ')
            hist0, hist1 = nested(before, 'mtp.completed_cycles_by_proposed_depth'), nested(after, 'mtp.completed_cycles_by_proposed_depth')
            require(len(hist0) == len(hist1) == 16, 'actual depth histogram shape differs')
            delta = [z - x for x, z in zip(hist0, hist1)]
            require(delta[4] >= 4 and not any(delta[5:]), 'at least four genuine trained cap4/R5 cycles required')
            for field, wanted in (('requests.submitted', 1), ('requests.completed', 1), ('requests.failed', 0), ('requests.cancelled', 0), ('metrics.metal_failures', 0)):
                require(nested(after, field) - nested(before, field) == wanted, 'actual request/Metal counters differ:' + field)
            result_info = {'output_u32le_sha256': hashlib.sha256(struct.pack('<64I', *result.tokens)).hexdigest(), 'finish_reason': result.done.reason.name, 'genuine_depth_histogram_delta': delta, 'cache_disposition': 'MISS', 'matched_prompt_tokens': 0, 'native_prefill_tokens': 2048, 'native_output_tokens': 64}
        except Exception as error:
            errors.append(type(error).__name__ + ':' + str(error))
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as error:
                    errors.append('close:' + str(error))
        terminal = len(processes) == 1 and processes[0].poll() == 0 and not processes[0].kill_called and group_gone(processes[0].pid)
        result_info.update({'native_processes': len(processes), 'returncodes': [p.poll() for p in processes], 'termination_signals': [{'terminate_called': p.terminate_called, 'kill_called': p.kill_called} for p in processes], 'backend_process_destroyed': terminal})
        require(terminal and not errors, 'clean one-process arm required:' + ';'.join(errors))
    proof = json.loads((directory / 'complete.json').read_text())
    wanted = {'schema': 'current-real-R5-capture-three-frame-proof-v1', 'complete': True, 'frames': 3, 'request_id': 1, 'generation': 1, 'physical_rows': 5, 'ordinal': 3, 'prior_successful_real_R5': 2, 'future_actual_R5_ordinal': 4, 'capture_enabled': name == 'candidate', 'copy_dispatches': 113 if name == 'candidate' else 0, 'roles': 113, 'logical_input_bytes': 4362240, 'aligned_input_bytes': 5046272, 'request_planes_per_frame': 134, 'initialized_lazy_arenas_per_frame': 216, 'PLE_cross_process_defined_only': True, 'undefined_tails_local_only': True, 'local_owner_binding_view_checks': True, 'capture_inputs_unchanged_through_commit_future': True, 'performance_claim': False}
    require(all(type(proof.get(k)) is type(v) and proof.get(k) == v for k, v in wanted.items()), 'actual strict three-frame proof metadata differs')
    require(proof['spill_bytes'] < 4 << 30 and proof['owner_allocation_delta'] <= 16 << 20, 'actual spill/capture owner bound exceeded')
    return result_info


def compare_bytes(left, right, expected_bytes):
    require(left.stat().st_size == right.stat().st_size == expected_bytes, 'physical plane extent differs')
    offset = 0
    with left.open('rb') as a, right.open('rb') as b:
        # Darwin F_NOCACHE=48, matching the native SDK macro used for writes.
        fcntl.fcntl(a.fileno(), 48, 1)
        fcntl.fcntl(b.fileno(), 48, 1)
        while offset < expected_bytes:
            x, y = a.read(min(65536, expected_bytes - offset)), b.read(min(65536, expected_bytes - offset))
            require(x and x == y, 'exact byte mismatch at chunk offset' + str(offset))
            offset += len(x)
    return offset


def compare_frames(command):
    a, b = (Path(command['arms'][n]['directory']) for n in ('control', 'candidate'))
    bytes_checked = planes_checked = 0
    for frame in FRAMES:
        left, right = json.loads((a / frame / 'frame.json').read_text()), json.loads((b / frame / 'frame.json').read_text())
        require(left == right, 'actual state/tape/output metadata differ:' + frame)
        for plane in left['planes']:
            filename = plane['file']
            require(Path(filename).name == filename, 'plane path must remain local')
            bytes_checked += compare_bytes(a / frame / filename, b / frame / filename, plane['bytes'])
            planes_checked += 1
    left, right = json.loads((a / 'raw-inputs.json').read_text()), json.loads((b / 'raw-inputs.json').read_text())
    require(len(left['roles']) == len(right['roles']) == 113, 'exact113 role inventory required')
    for x, y in zip(left['roles'], right['roles']):
        x, y = dict(x), dict(y)
        require(x.pop('file') is None, 'capture-off arm must not export raw payload')
        filename = y.pop('file')
        require(x == y and Path(filename).name == filename and (b / filename).stat().st_size == y['logical_bytes'], 'captured actual role/source/format/file extent differs')
    return {'three_frames_bit_exact': True, 'bytes_compared': bytes_checked, 'planes_compared': planes_checked, 'captured_actual_roles': 113}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command', type=Path, required=True)
    parser.add_argument('--command-sha256', required=True)
    parser.add_argument('--run-root-gpu', action='store_true')
    args = parser.parse_args()
    require(args.run_root_gpu, 'explicit Root-only GPU and capture-data invocation required')
    require(sha(args.command) == args.command_sha256, 'externally pinned command differs')
    command = json.loads(args.command.read_text())
    require(command['schema'] == 'Root-current-trained-R5-raw-capture-three-frame-command-v1' and command['context'] == 2048 and command['output_tokens'] == 64 and command['request_id'] == command['generation'] == command['requests'] == 1, 'exact one request per process required')
    require(command['tokens'] == str(TOKEN_FILE) and command['tokens_file_sha256'] == TOKEN_SHA and command['tokens_u32le_sha256'] == TOKEN_U32_SHA and command['library_sha256'] == LIBRARY, 'exact static current artifact/token pins required')
    for path, digest in command['program_pins'].items():
        require(sha(path) == digest, 'Root-pinned program source differs')
    build = Path(command['build'])
    require(sha(build / 'splash-flash') == command['exe_sha256'] and sha(build / 'splash.metallib') == LIBRARY, 'fresh capture-worker artifact differs')
    report = Path(command['report'])
    require(not any(Path(str(report) + suffix).exists() for suffix in ('', '.failure.json', '.writing', '.partial')), 'fresh capture QA report required')
    environment = command['environment']
    require(environment['SPLASH_FLASH_MTP'] == '1' and environment['SPLASH_FLASH_MTP_DRAFT_DEPTH'] == '4' and environment['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'] == '1', 'current trained fixed4/R5 flags required')
    for field in ('SPLASH_FLASH_BATCH', 'SPLASH_FLASH_BATCH_MTP', 'SPLASH_FLASH_BATCH_PREFILL', 'SPLASH_FLASH_BATCH_MTP_PREFILL', 'SPLASH_FLASH_DIAG_R5_TARGET_STAGE_SEP22'):
        require(environment.get(field, '0') == '0', 'excluded batch/profiling mode:' + field)
    require(sha(TOKEN_FILE) == TOKEN_SHA, 'Root exact token-file witness differs')
    tokens = json.loads(TOKEN_FILE.read_text())
    require(type(tokens) is list and len(tokens) == 2048 and all(type(x) is int and 0 <= x < 248320 for x in tokens), 'valid exact2048-token fixture required')
    require(hashlib.sha256(struct.pack('<2048I', *tokens)).hexdigest() == TOKEN_U32_SHA, 'Root canonical token witness differs')
    with (ROOT / 'build/splash-tuning-gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = run_arm(command, 'control', tokens)
        candidate = run_arm(command, 'candidate', tokens)
        require(control['output_u32le_sha256'] == candidate['output_u32le_sha256'] and control['finish_reason'] == candidate['finish_reason'] and control['genuine_depth_histogram_delta'] == candidate['genuine_depth_histogram_delta'], 'actual trained output/acceptance sequence differs')
        compared = compare_frames(command)
    result = {'schema': 'Root-current-trained-R5-raw-capture-three-frame-QA-v1', 'pass': True, 'capture_QA_complete': True, 'performance_claim': False, 'trained_head_numerical_oracle_claimed': False, 'worker_source_parent': '4bb7b637c2b6b60520159ad3724a8d9e3867c7c538dd8d260caa4fc7d184769a', 'exe_sha256': command['exe_sha256'], 'library_sha256': LIBRARY, 'command_sha256': args.command_sha256, 'control': control, 'candidate': candidate, 'comparison': compared, 'backend_processes_destroyed': True}
    with Path(str(report) + '.writing').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    Path(str(report) + '.writing').rename(report)
    print(json.dumps({'pass': True, 'report': str(report), 'capture_QA_not_performance': True}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Root-only execution failures stay failclosed and remain reviewable.
        if '--run-root-gpu' in sys.argv and '--command' in sys.argv:
            try:
                command_path = Path(sys.argv[sys.argv.index('--command') + 1])
                command = json.loads(command_path.read_text())
                failure = Path(str(command['report']) + '.failure.json')
                with failure.open('x') as stream:
                    json.dump({'schema': 'Root-current-trained-R5-raw-capture-QA-failure-v1', 'pass': False, 'capture_QA_complete': False, 'performance_claim': False, 'error': type(error).__name__ + ':' + str(error), 'clean_backend_terminal_not_asserted_without_success': True}, stream, indent=2)
                    stream.write('\n')
            except Exception:
                pass
        print(type(error).__name__ + ':' + str(error), file=sys.stderr)
        raise SystemExit(2)
