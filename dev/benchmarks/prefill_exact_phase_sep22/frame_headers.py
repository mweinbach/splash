#!/usr/bin/env python3
"""Read bounded native PREF frame headers; seek over every tensor payload.

FileIO is unbuffered: no Python read-ahead may consume payload bytes. This
parser neither hashes nor imports tensor data. Root may independently inspect
payload values with separate tools if needed.
"""
import argparse
import io
import json
from pathlib import Path
import struct

MAGIC = b'SPLASHPREFEX\x01\x00\x00\x00'
LIMIT = 4 << 30
WIDTH = {1: 2, 2: 4, 3: 8, 4: 4}
TYPE = {1: 'BF16', 2: 'F32', 3: 'I64', 4: 'U32'}


def parse_headers(handle, file_bytes):
    if file_bytes < 32 or file_bytes > LIMIT:
        raise ValueError('frame physical extent outside bounds')
    header_reads = 0

    def read(count, maximum):
        nonlocal header_reads
        if count < 0 or count > maximum or handle.tell() + count > file_bytes:
            raise ValueError('bounded header declaration exceeds frame')
        result = handle.read(count)
        if len(result) != count:
            raise ValueError('truncated frame header')
        header_reads += count
        if header_reads > (128 << 10):
            raise ValueError('aggregate header bytes exceed bound')
        return result

    def number():
        return struct.unpack('<Q', read(8, 8))[0]

    def text(maximum):
        count = number()
        return read(count, maximum).decode('utf-8', errors='strict')

    if read(16, 16) != MAGIC:
        raise ValueError('frame magic/version differs')
    metadata = json.loads(text(65536))
    if not isinstance(metadata, dict):
        raise ValueError('frame metadata must be a dictionary')
    count = number()
    if count == 0 or count > 134:
        raise ValueError('plane count outside bound')
    planes = []
    names = set()
    for _ in range(count):
        declaration_begin = handle.tell()
        label = text(256)
        dtype, live_bytes, physical_bytes = number(), number(), number()
        if not label or label in names or dtype not in WIDTH or live_bytes > physical_bytes:
            raise ValueError('plane declaration invalid')
        if physical_bytes > LIMIT or physical_bytes > file_bytes - handle.tell():
            raise ValueError('plane payload extent exceeds physical frame')
        payload_begin = handle.tell()
        # The only operation touching the tensor extent is a seek, never read.
        handle.seek(physical_bytes, io.SEEK_CUR)
        planes.append({'label': label, 'type': TYPE[dtype], 'word_bytes': WIDTH[dtype],
                       'live_bytes': live_bytes, 'physical_bytes': physical_bytes,
                       'declaration_begin': declaration_begin,
                       'payload_begin': payload_begin, 'payload_end': handle.tell()})
        names.add(label)
    if handle.tell() != file_bytes:
        raise ValueError('unexpected trailing frame extent')
    return {'pass': True, 'metadata': metadata, 'file_bytes': file_bytes,
            'planes': planes, 'header_bytes_read': header_reads,
            'tensor_payload_bytes_read': 0, 'tensor_payload_bytes_hashed': 0}


def locate(result, offset):
    if offset < 0 or offset >= result['file_bytes']:
        raise ValueError('requested byte offset outside frame')
    for plane in result['planes']:
        if plane['payload_begin'] <= offset < plane['payload_end']:
            relative = offset - plane['payload_begin']
            return {'region': 'tensor_payload', 'plane': plane['label'],
                    'type': plane['type'], 'absolute_byte': offset,
                    'plane_byte': relative, 'word_index': relative // plane['word_bytes'],
                    'byte_in_word': relative % plane['word_bytes']}
        if plane['declaration_begin'] <= offset < plane['payload_begin']:
            return {'region': 'plane_header', 'plane': plane['label'], 'absolute_byte': offset}
    return {'region': 'frame_header', 'absolute_byte': offset}


def self_test():
    meta = {'length': 2064, 'capacity': 4096, 'consumed_rows': 16,
            'logit_rows': 16, 'greedy_rows': 16}
    packed = json.dumps(meta, separators=(',', ':')).encode()
    data = bytearray(MAGIC)
    data += struct.pack('<Q', len(packed)) + packed + struct.pack('<Q', 1)
    label = b'hidden'
    data += struct.pack('<Q', len(label)) + label + struct.pack('<QQQ', 1, 8, 8)
    payload_begin = len(data)
    data += b'\xa5' * 8  # Synthetic bytes only; parser must never read these.

    class GuardedIO(io.BytesIO):
        def read(self, count=-1):
            if count < 0 or self.tell() < len(data) and self.tell() + count > payload_begin:
                raise AssertionError('parser attempted a tensor payload read')
            return super().read(count)

    result = parse_headers(GuardedIO(data), len(data))
    assert payload_begin == 153
    assert locate(result, 153) == {'region': 'tensor_payload', 'plane': 'hidden',
                                  'type': 'BF16', 'absolute_byte': 153, 'plane_byte': 0,
                                  'word_index': 0, 'byte_in_word': 0}
    assert locate(result, 152)['region'] == 'plane_header'
    rejected = False
    try:
        parse_headers(GuardedIO(data), len(data) - 1)
    except ValueError:
        rejected = True
    assert rejected
    return {'pass': True, 'checks': ['unbuffered_header_only_contract',
                                    'payload_read_trap', 'literal_r16_hidden_offset153',
                                    'truncated_extent_rejected'],
            'gpu_executed': False, 'tensor_payload_bytes_read': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame', nargs='?')
    parser.add_argument('--offset', type=int)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    if not args.frame:
        parser.error('a frame or --self-test is required')
    path = Path(args.frame)
    file_bytes = path.stat().st_size
    with path.open('rb', buffering=0) as handle:
        result = parse_headers(handle, file_bytes)
    if args.offset is not None:
        result['requested_offset'] = locate(result, args.offset)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
