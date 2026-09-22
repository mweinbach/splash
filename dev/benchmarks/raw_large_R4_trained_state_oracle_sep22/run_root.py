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
CANDIDATE = 'SPLASH_FLASH_RAW_LARGE_R4_GUARD_PAIR_SEP22'
PROOF = 'SPLASH_FLASH_DIAG_RAW_LARGE_R4_TRAINED_STATE_SEP22'
DIRECTORY = 'SPLASH_FLASH_DIAG_RAW_LARGE_R4_STATE_DIRECTORY_SEP22'
FRAMES = ('third_pending', 'third_resolved', 'next_actual_target')


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
    environment[CANDIDATE], environment[PROOF], environment[DIRECTORY] = ('0' if name == 'control' else '1'), '1', str(directory)
    processes = []
    runtime = None
    errors = []
    result_info = {}
    with log_path.open('x') as log:
        def create():
            require(not processes, 'automatic native restart forbidden')
            process = subprocess.Popen([str(Path(arm['build']) / 'splash-flash'), 'serve-flash-native', command['package'], '16384', 'auto'], cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0, start_new_session=True)
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
            require(nested(before, 'mtp.singleton_maximum_draft_tokens') == 3, 'actual singleton cap3 required')
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
            require(delta[3] >= 4 and not any(delta[4:]), 'at least four genuine trained cap4/R5 cycles required')
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
    wanted = {'schema': 'genuine-trained-R4-three-boundary-complete-v1', 'complete': True, 'frames': 3, 'request_id': 1, 'generation': 1, 'depth': 3, 'physical_rows': 4, 'ordinal': 3, 'prior_successful_R4_calls': 2, 'main_physical_planes': 134, 'trained_head_physical_planes': 5, 'known_lazy_physical_arenas': 216, 'new_backend_owners': 0, 'input_copy_dispatches': 0, 'diagnostic_reservation_released': True, 'undefined_tails_local_only': True, 'owner_views_and_lazy_redzones_valid': True, 'worker_defined_carry_and_offsets_included': True, 'ordinary26_54_or_task_or_performance_claim': False, 'selected_legacy_Q4_graph_calls': 26, 'selected_legacy_Q4_graph_rows': 104, 'selected_new_pair_graph_calls': 87 if name == 'candidate' else 0, 'selected_new_pair_graph_rows': 348 if name == 'candidate' else 0}
    require(all(type(proof.get(k)) is type(v) and proof.get(k) == v for k, v in wanted.items()), 'strict actual genuine trained three-boundary proof metadata differs')
    require(proof['spill_bytes'] <= proof['whole_preflight_bytes'] < 4 << 30, 'actual/source whole spill bound exceeded')
    result_info['actual_native_proof'] = proof
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
    require(json.loads((a/'policy.json').read_text()) == json.loads((b/'policy.json').read_text()), 'exact canonical measured policy or whole preflight differs')
    bytes_checked = planes_checked = 0
    for frame in FRAMES:
        left, right = json.loads((a/frame/'frame.json').read_text()), json.loads((b/frame/'frame.json').read_text())
        for arm, value in [('control', left), ('candidate', right)]:
            require(value['legacy_Q4_graph_calls'] == 26 and value['legacy_Q4_graph_rows'] == 104, 'actual typed legacy counter differs')
            require(value['new_pair_graph_calls'] == (87 if arm == 'candidate' else 0) and value['new_pair_graph_rows'] == (348 if arm == 'candidate' else 0), 'actual typed newpair counter differs')
            for key in ['legacy_Q4_graph_calls', 'legacy_Q4_graph_rows', 'new_pair_graph_calls', 'new_pair_graph_rows']:
                value.pop(key)
        require(left == right, 'actual main/head/carry/output/offset metadata differ:' + frame)
        labels = [p['label'] for p in left['planes']]
        require(sum(x.startswith('main.') for x in labels) == 134 and sum(x.startswith('trainedhead.') for x in labels) == 5 and sum(x.startswith('tape.gdn.') for x in labels) == 216, 'complete actual main/head/lazy inventory missing')
        require('worker.defined_fold_hidden' in labels, 'actual owned foldcarry missing')
        for plane in left['planes']:
            filename = plane['file']
            require(Path(filename).name == filename, 'plane path must remain local')
            bytes_checked += compare_bytes(a/frame/filename, b/frame/filename, plane['bytes'])
            planes_checked += 1
    return {'three_boundaries_bit_exact': True, 'bytes_compared': bytes_checked, 'planes_compared': planes_checked, 'main_planes_per_boundary': 134, 'trained_head_planes_per_boundary': 5, 'known_lazy_arenas_per_boundary': 216, 'actual_resolved_head_and_foldcarry_included': True}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command', type=Path, required=True)
    parser.add_argument('--command-sha256', required=True)
    parser.add_argument('--run-root-gpu', action='store_true')
    args = parser.parse_args()
    require(args.run_root_gpu, 'explicit Root-only GPU and capture-data invocation required')
    require(sha(args.command) == args.command_sha256, 'externally pinned command differs')
    command = json.loads(args.command.read_text())
    require(command['schema'] == 'Root-current-trained-R4-three-boundary-command-v1' and command['context'] == 2048 and command['output_tokens'] == 64 and command['request_id'] == command['generation'] == command['requests'] == 1, 'exact one request per process required')
    require(command['tokens'] == str(TOKEN_FILE) and command['tokens_file_sha256'] == TOKEN_SHA and command['tokens_u32le_sha256'] == TOKEN_U32_SHA , 'exact static current artifact/token pins required')
    for path, digest in command['program_pins'].items():
        require(sha(path) == digest, 'Root-pinned program source differs')
    for name in ['control', 'candidate']:
        arm = command['arms'][name]
        build = Path(arm['build'])
        require(sha(build/'splash-flash') == arm['exe_sha256'] and sha(build/'splash.metallib') == arm['library_sha256'], 'fresh reviewed diagnostic artifact differs:' + name)
    report = Path(command['report'])
    require(not any(Path(str(report) + suffix).exists() for suffix in ('', '.failure.json', '.writing', '.partial')), 'fresh capture QA report required')
    environment = command['environment']
    require(environment['SPLASH_FLASH_MTP'] == '1' and environment['SPLASH_FLASH_MTP_DRAFT_DEPTH'] == '3', 'current trained fixed4/R5 flags required')
    for field in ('SPLASH_FLASH_DIAG_R5_TARGET_STAGE_SEP22',):
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
    result = {'schema': 'Root-current-trained-R4-three-boundary-QA-v1', 'pass': True, 'trained_three_boundary_QA_complete': True, 'performance_claim': False, 'ordinary26_54_claim': False, 'normal_producer_bindings': command['normal_producers'], 'private_oracle_bindings': {n: {k: command['arms'][n][k] for k in ['build', 'exe_sha256', 'library_sha256']} for n in ['control', 'candidate']}, 'command_sha256': args.command_sha256, 'measured_policy_command_sha256': command['measured_policy_command_sha256'], 'control': control, 'candidate': candidate, 'comparison': compared, 'backend_processes_destroyed': True}
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
                    json.dump({'schema': 'Root-current-trained-R4-three-boundary-QA-failure-v1', 'pass': False, 'capture_QA_complete': False, 'performance_claim': False, 'error': type(error).__name__ + ':' + str(error), 'clean_backend_terminal_not_asserted_without_success': True}, stream, indent=2)
                    stream.write('\n')
            except Exception:
                pass
        print(type(error).__name__ + ':' + str(error), file=sys.stderr)
        raise SystemExit(2)
