#!/usr/bin/env python3
"""Extract verbatim frozen native guard decisions, never model data or hashes.

The generated C++ supplies metadata-only type adapters separately. Function
bodies and guard prefixes are copied unchanged; only declaration scaffolding
and no-op unused-variable casts are synthesized. Source-anchor drift is fatal.
"""
from pathlib import Path
import argparse
import json


def braces(text, opening):
    depth, quote, line_comment, block_comment, escape = 0, None, False, False, False
    i = opening
    while i < len(text):
        c = text[i]
        nxt = text[i + 1:i + 2]
        if line_comment:
            if c == '\n': line_comment = False
        elif block_comment:
            if c == '*' and nxt == '/': block_comment = False; i += 1
        elif quote:
            if escape: escape = False
            elif c == '\\': escape = True
            elif c == quote: quote = None
        elif c == '/' and nxt == '/': line_comment = True; i += 1
        elif c == '/' and nxt == '*': block_comment = True; i += 1
        elif c in ('"', "'"): quote = c
        elif c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return i + 1
        i += 1
    raise ValueError('Unclosed frozen source body')


def function(text, anchor):
    if text.count(anchor) != 1: raise ValueError('Frozen source anchor drift: ' + anchor)
    begin = text.index(anchor)
    opening = text.index('{', begin)
    return text[begin:braces(text, opening)], begin


def prefix(text, anchor, stop):
    full, begin = function(text, anchor)
    if full.count(stop) != 1: raise ValueError('Frozen guard stop drift: ' + stop)
    return full[:full.index(stop)], begin


def build(store, buckets):
    chunks, records = [], []
    def add(name, text, anchor):
        value, start = function(text, anchor)
        records.append({'name': name, 'line': text[:start].count('\n') + 1, 'verbatim_characters': len(value)})
        return value
    chunks.append('#pragma once\n// Generated verbatim guard baseline: source-only; no backend/model.\nnamespace frozen_baseline {\n')
    chunks.append(add('fail', store, '[[noreturn]] void fail('))
    chunks.append(add('requireGeometry', buckets, 'void requireGeometry('))
    chunks.append(add('moEBucketJobCapacity', buckets, 'uint32_t moEBucketJobCapacity('))
    chunks.append('namespace metal=::metadata_fake;\nusing FlashMoEBlockedScratch=::metadata_fake::Scratch;\nusing FlashMoEBlockedTile=::metadata_fake::Tile;\n')
    # The extracted capacity helper's declarations require its actual enums and
    # constants before definition; the harness defines those without decisions.
    chunks[2:2] = ['using FlashMoEBlockedTile=::metadata_fake::Tile;\n']
    chunks.append(add('requireBytes', store, 'void requireBytes('))
    chunks.append(add('disjoint', store, 'void disjoint('))
    chunks.append('inline bool flashMoEDirectAEnabled(){return ::metadata_fake::configuration->directA;}\n'
                  'inline FlashMoEBlockedTile flashMoEBlockedTile(uint32_t,bool){return ::metadata_fake::configuration->wideM64?FlashMoEBlockedTile::M64N64:FlashMoEBlockedTile::M32N64;}\n')
    chunks.append(add('allRowsScratch', store, 'void allRowsScratch('))
    layer = add('layer', store, '  const Layer &layer(uint32_t index) const')
    immutable = add('immutableDisjoint', store, '  void immutableDisjoint(')
    chunks.append('struct Impl {using Layer=::metadata_fake::Layer;std::array<Layer,48> layers;::metadata_fake::Metadata metadata;bool compactR4Verify=true;\n')
    chunks.append(layer); chunks.append(immutable); chunks.append('};\n')
    chunks.append('class FlashInt8ExpertStore {public: Impl *impl_=nullptr;\n'
                  'bool gatheredMPPEnabled()const{return ::metadata_fake::configuration->gather;}\n'
                  'uint32_t gatheredMPPMaximumRows()const{return ::metadata_fake::configuration->gatherCap;}\n'
                  'bool compactNativeR4VerifyEnabled()const;\n'
                  'void addCompactNativeR4VerifyPack(metal::CommandGraph &,uint32_t,metal::MetalBuffer,metal::MetalBuffer,const FlashMoEBlockedScratch &,metal::MetalBuffer,uint32_t,uint32_t)const;\n'
                  'void addGateUp(metal::CommandGraph &,uint32_t,const FlashMoEBlockedScratch &,metal::MetalBuffer,uint32_t,FlashMoEBlockedTile,uint32_t)const;\n'
                  'void addDownScatter(metal::CommandGraph &,uint32_t,const FlashMoEBlockedScratch &,metal::MetalBuffer,uint32_t,FlashMoEBlockedTile,uint32_t)const;\n};\n'
                  'namespace compact_native_r4_verify_sep22 {inline bool requested(){return ::metadata_fake::configuration->requested;}}\n')
    chunks.append(add('compactEnabled', store, 'bool FlashInt8ExpertStore::compactNativeR4VerifyEnabled() const'))
    starts = [('pack', 'void FlashInt8ExpertStore::addCompactNativeR4VerifyPack(', '  graph.add("expert_r4_compact_native_sep22_plan"'),
              ('gate', 'void FlashInt8ExpertStore::addGateUp(', '  const auto p = allRowsParams('),
              ('down', 'void FlashInt8ExpertStore::addDownScatter(', '  const uint32_t routes = rows * selections;')]
    pack_value = None
    for name, anchor, stop in starts:
        value, start = prefix(store, anchor, stop)
        records.append({'name': name + '_guard_prefix', 'line': store[:start].count('\n') + 1, 'verbatim_characters': len(value)})
        if name == 'pack': pack_value = value
        chunks.append(value + '  (void)graph; (void)layer;\n}\n')
    # The new wrapper copies only original outer policy/index/inventory. It
    # then calls the actual portable shipping routine, passing original guards.
    stop = '  allRowsScratch('
    if pack_value.count(stop) != 1: raise ValueError('Original Pack outer-prefix drift')
    outer = pack_value[:pack_value.index(stop)]
    body = outer[outer.index('{') + 1:]
    chunks.append('inline void validateBundled(FlashInt8ExpertStore &store,metal::CommandGraph &graph,uint32_t index,metal::MetalBuffer input,metal::MetalBuffer ids,\n'
                  'const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections){\n'
                  'auto *impl_=store.impl_;\n'
                  'const auto compactNativeR4VerifyEnabled=[&]{return store.compactNativeR4VerifyEnabled();};\n')
    chunks.append(body)
    chunks.append('::splash::flash::compact_r4_preflight_sep22::validateComplete(s,input,ids,layer.ranks,diagnostics,rows,selections,\n'
                  '[](const auto &scratch,auto diag,uint32_t r,uint32_t k){allRowsScratch(scratch,diag,r,FlashMoEBlockedTile::M16N64,k);},\n'
                  '[](const auto &buffer,uint64_t bytes){requireBytes(buffer,bytes);},\n'
                  '[](const auto &a,const auto &b){disjoint(a,b);},\n'
                  '[&](const auto &buffer){impl_->immutableDisjoint(buffer);});\n(void)graph;\n}\n}\n')
    return '\n'.join(chunks), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Sealed v1b source root')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    store = (args.source / 'runtime/flash/FlashInt8ExpertStore.mm').read_text()
    buckets = (args.source / 'runtime/flash/FlashMoEBuckets.cpp').read_text()
    generated, records = build(store, buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(json.dumps({'generated': str(args.output), 'verbatim_source_blocks': records,
        'gpu_work': False, 'backend_created': False, 'model_or_capture_payload_read_or_hashed': False}))


if __name__ == '__main__': main()
