"""Root-only native transport proof at the private post-bulk/pre-tail boundary."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.runtime import Deadline, GenerationRequest, MultiplexedRuntime, RequestFailed
from server import protocol as wire


def check(value, message):
    if not value:
        raise AssertionError(message)


def nested(value, path):
    for key in path.split('.'):
        value = value[key]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-root-gpu', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    binary = Path(config['binary'])
    tokens_path = ROOT / 'build/release/flash/prefill4k-fixture/code2048.tokens.json'
    tokens = tuple(json.loads(tokens_path.read_text()))
    token_sha = hashlib.sha256(struct.pack('<' + 'I' * len(tokens), *tokens)).hexdigest()
    check(len(tokens) == 2048 and token_sha == '0a383d21f5c784b0616d589847ca6cb04c69bf654729f2344e2b916e542f36b4', 'canonical token witness differs')
    check(config['resolved_environment']['SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS'] == '500', 'QA pause must be bounded500ms')
    command = [str(binary), 'serve-flash-native', str(ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'), '16384', 'auto']
    report = {'schema': 'splash-teacher-worker-native-recovery-sep22-v1', 'valid': False,
              'gpu_executed': False, 'command': command, 'tokens_u32le_sha256': token_sha,
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'metallib_sha256': hashlib.sha256(binary.with_name('splash.metallib').read_bytes()).hexdigest(),
              'qa_pause_ms': 500, 'cases': [], 'same_id_reuse_scope': 'after the cancelled call is terminal; concurrent replacement is covered separately by extracted CPU scheduler fixture'}
    if not args.run_root_gpu:
        print(json.dumps(report, indent=2)); return
    check(not args.output.exists(), 'choose a fresh report')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logpath = args.output.with_suffix('.native.log')
    environment = {k: v for k, v in os.environ.items() if not k.startswith('SPLASH_FLASH_')}
    environment.update(config['resolved_environment'])
    processes = []
    runtime = None
    lock = (ROOT / 'build/splash-tuning-gpu.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = logpath.open('x')

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def create_process():
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=log, bufsize=0, start_new_session=True)
        processes.append(process)
        return process

    def status():
        return json.loads(runtime.status(timeout=3).json)

    def idle():
        limit = time.monotonic() + 30
        while time.monotonic() < limit:
            s = status()
            check(s['ready'] and s['memory_pressure'] == 'normal', 'worker unhealthy')
            q = s['scheduler']
            if not q['active_requests'] and not q['queued'] and not q['command_in_flight']:
                return s
            time.sleep(.02)
        raise TimeoutError('worker did not become idle')

    def submit(seconds=60):
        return runtime.submit(GenerationRequest(tokens, 8, Deadline.after(seconds)))

    def boundary(call, old):
        limit = time.monotonic() + 30
        while time.monotonic() < limit:
            s = status()
            qa = nested(s, 'mtp.singleton_teacher_bulk_qa')
            count = nested(s, 'mtp.singleton_teacher_bulk.completed_bulk_commands')
            if qa['boundary_open'] and count == old + 1:
                check(qa['completed_pairs_at_boundary'] == 1920 and qa['pending_tail_pairs_at_boundary'] == 127, 'wrong bulk boundary')
                return s
            check(not call.done, 'call ended before observable bulk boundary')
            time.sleep(.003)
        raise TimeoutError('bulk boundary was not observed')

    def recover(label):
        before = idle()
        start = time.monotonic()
        call = submit()
        at = boundary(call, nested(before, 'mtp.singleton_teacher_bulk.completed_bulk_commands'))
        latency = time.monotonic() - start
        result = call.result(timeout=30)
        check(result.done.reason in (wire.FinishReason.STOP, wire.FinishReason.LENGTH) and result.tokens, 'recovery request did not complete')
        after = idle()
        check(nested(after, 'mtp.singleton_teacher_bulk.completed_tail_commands') - nested(before, 'mtp.singleton_teacher_bulk.completed_tail_commands') == 1, 'valid recovery omitted its127tail')
        report['cases'].append({'name': label, 'request_id': call.request_id, 'boundary_latency_seconds': latency,
                                'boundary': at, 'finish_reason': result.done.reason.name,
                                'completion_tokens': len(result.tokens), 'after': after})
        save()
        return latency

    try:
        report['gpu_executed'] = True
        runtime = MultiplexedRuntime(process_factory=create_process, startup_timeout=300, io_timeout=5)
        runtime.wait_ready(timeout=300)
        report['initial_status'] = idle(); save()
        recover('cold_valid_request')
        warm_latency = recover('warm_valid_request')
        before = idle(); call = submit()
        at = boundary(call, nested(before, 'mtp.singleton_teacher_bulk.completed_bulk_commands'))
        check(call.cancel(), 'cancellation was not written')
        result = call.result(timeout=30)
        check(result.done.reason is wire.FinishReason.CANCELLED and not result.tokens, 'request did not cancel before emission')
        after = idle()
        check(nested(after, 'mtp.singleton_teacher_bulk.completed_tail_commands') == nested(before, 'mtp.singleton_teacher_bulk.completed_tail_commands'), 'cancelled request submitted127tail')
        report['cases'].append({'name': 'cancel_at_postbulk_boundary', 'request_id': call.request_id, 'boundary': at, 'after': after}); save()
        # Reuse only after terminal cancellation, never overwrite a live client call.
        runtime._request_ids = itertools.count(call.request_id)
        recover('same_id_after_terminal_cancel')
        before = idle()
        duration = warm_latency + .25
        start = time.monotonic(); call = submit(duration)
        at = boundary(call, nested(before, 'mtp.singleton_teacher_bulk.completed_bulk_commands'))
        remaining = duration - (time.monotonic() - start)
        check(0 < remaining < .5, 'deadline did not fall inside the500ms pause')
        try:
            call.result(timeout=30)
        except RequestFailed as error:
            check(error.code == b'deadline_exceeded', 'unexpected deadline error')
            terminal = {'code': error.code.decode(), 'message': str(error)}
        else:
            raise AssertionError('deadline request completed instead of expiring')
        after = idle()
        check(nested(after, 'mtp.singleton_teacher_bulk.completed_tail_commands') == nested(before, 'mtp.singleton_teacher_bulk.completed_tail_commands'), 'expired request submitted127tail')
        report['cases'].append({'name': 'deadline_at_postbulk_boundary', 'request_id': call.request_id,
                                'deadline_seconds': duration, 'remaining_at_boundary_seconds': remaining,
                                'boundary': at, 'terminal': terminal, 'after': after}); save()
        recover('valid_after_deadline')
        check(len(processes) == 1 and processes[0].poll() is None, 'native process restarted or exited')
        report['valid'] = True
        report['native_process_count'] = len(processes)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
        report['unloaded'] = all(process.poll() is not None for process in processes)
        report['returncodes'] = [process.returncode for process in processes]
        save(); log.close(); lock.close()
    print(json.dumps({'report': str(args.output), 'valid': report['valid'], 'unloaded': report['unloaded']}))


if __name__ == '__main__':
    main()
