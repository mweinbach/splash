from pathlib import Path
import argparse


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError(f'memory kernel source drift: {before!r}')
    return text.replace(before, after)


def generate(destination):
    source = Path('dev/benchmarks/prefill4k_int8tiles/candidate.metal').read_text()
    source = source.replace('prefill4k_int8tiles_', 'prefill_moe_sep21_memory_')
    source = replace(source, 'template <ushort M, ushort SG>', 'template <ushort M, ushort SG, ushort K = 0, bool Static = false>', 3)
    source = replace(source, 'constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),\n      false, true, false, matmul2d_descriptor::mode::multiply);',
                     'constexpr auto descriptor = matmul2d_descriptor(M, N, K ? int(K) : static_cast<int>(dynamic_extent),\n      false, true, false, K ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply);', 2)
    source = replace(source, '  operation.run(a, g, gd); operation.run(a, u, ud);', '''  if constexpr (K) {
#pragma unroll
    for (ushort i = 0; i < gd.get_capacity(); ++i) {
      if (gd.is_valid_element(i)) { gd[i] = 0.0f; ud[i] = 0.0f; }
    }
    for (uint k = 0; k < 2560; k += K) {
      if constexpr (Static) {
        if (valid_rows == M) {
          auto aa = tensor(input + ulong(begin) * 2560 + k, extents<int, K, M>{}, array<int, 2>{1, 2560});
          auto gg = tensor(gate + (ulong(rank) * 640 + column) * 2560 + k, extents<int, K, N>{}, array<int, 2>{1, 2560});
          auto uu = tensor(up + (ulong(rank) * 640 + column) * 2560 + k, extents<int, K, N>{}, array<int, 2>{1, 2560});
          operation.run(aa, gg, gd); operation.run(aa, uu, ud);
        } else {
          auto aa = a.slice(k, 0); auto gg = g.slice(k, 0); auto uu = u.slice(k, 0);
          operation.run(aa, gg, gd); operation.run(aa, uu, ud);
        }
      } else {
        auto aa = a.slice(k, 0); auto gg = g.slice(k, 0); auto uu = u.slice(k, 0);
        operation.run(aa, gg, gd); operation.run(aa, uu, ud);
      }
    }
  } else if constexpr (Static) {
    if (valid_rows == M) {
      auto aa = tensor(input + ulong(begin) * 2560, extents<int, 2560, M>{}, array<int, 2>{1, 2560});
      auto gg = tensor(gate + (ulong(rank) * 640 + column) * 2560, extents<int, 2560, N>{}, array<int, 2>{1, 2560});
      auto uu = tensor(up + (ulong(rank) * 640 + column) * 2560, extents<int, 2560, N>{}, array<int, 2>{1, 2560});
      operation.run(aa, gg, gd); operation.run(aa, uu, ud);
    } else { operation.run(a, g, gd); operation.run(a, u, ud); }
  } else { operation.run(a, g, gd); operation.run(a, u, ud); }''')
    source = replace(source, '  operation.run(a, b, dot);', '''  if constexpr (K) {
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i)
      if (dot.is_valid_element(i)) dot[i] = 0.0f;
    for (uint k = 0; k < 640; k += K) {
      if constexpr (Static) {
        if (valid_rows == M) {
          auto aa = tensor(input + ulong(begin) * 640 + k, extents<int, K, M>{}, array<int, 2>{1, 640});
          auto bb = tensor(weights + (ulong(rank) * 2560 + column) * 640 + k, extents<int, K, N>{}, array<int, 2>{1, 640});
          operation.run(aa, bb, dot);
        } else {
          auto aa = a.slice(k, 0); auto bb = b.slice(k, 0); operation.run(aa, bb, dot);
        }
      } else {
        auto aa = a.slice(k, 0); auto bb = b.slice(k, 0); operation.run(aa, bb, dot);
      }
    }
  } else if constexpr (Static) {
    if (valid_rows == M) {
      auto aa = tensor(input + ulong(begin) * 640, extents<int, 640, M>{}, array<int, 2>{1, 640});
      auto bb = tensor(weights + (ulong(rank) * 2560 + column) * 640, extents<int, 640, N>{}, array<int, 2>{1, 640});
      operation.run(aa, bb, dot);
    } else { operation.run(a, b, dot); }
  } else { operation.run(a, b, dot); }''')
    # SG1 must use the distinct execution_simdgroup type to admit a register-control comparison.
    source = replace(source, '  matmul2d<descriptor, execution_simdgroups<SG>> operation;',
                     '  using Scope = conditional_t<SG == 1, execution_simdgroup, execution_simdgroups<SG>>;\n  matmul2d<descriptor, Scope> operation;', 2)
    source = source.replace('#define PREFILL4K_INT8TILES_GATE(NAME, M, SG)', '#define PREFILL4K_INT8TILES_GATE(NAME, M, SG, K, STATIC)')
    source = source.replace('#define PREFILL4K_INT8TILES_DOWN(NAME, M, SG)', '#define PREFILL4K_INT8TILES_DOWN(NAME, M, SG, K, STATIC)')
    source = source.replace('prefill_moe_sep21_memory_gate<M, SG>(a,', 'prefill_moe_sep21_memory_gate<M, SG, K, STATIC>(a,')
    source = source.replace('prefill_moe_sep21_memory_down<M, SG>(a,', 'prefill_moe_sep21_memory_down<M, SG, K, STATIC>(a,')
    begin = source.index('// M128 requires independently generated')
    end = source.index('#undef PREFILL4K_INT8TILES_GATE', begin)
    definitions = []
    for kind, k, sg, static in [('whole',0,1,False),('whole',0,2,False),('static',0,1,True),('static',0,2,True),('fixed',64,1,True),('fixed',128,1,True),('fixed',128,2,True)]:
        for phase, macro in [('gate_up','GATE'),('down_scatter','DOWN')]:
            name = f'prefill_moe_sep21_memory_{kind}_{phase}_m32_n64_k{k}_sg{sg}'
            definitions.append(f'PREFILL4K_INT8TILES_{macro}({name}, 32, {sg}, {k}, {str(static).lower()})')
    source = source[:begin] + '\n'.join(definitions) + '\n' + source[end:]
    source = source.replace('// Private hit-only tile experiment.', '// Private low-SIMD and fixed-K expert matmul experiment.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    generate(parser.parse_args().destination)
