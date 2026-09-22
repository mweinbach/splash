#!/usr/bin/env python3
"""Generate isolated exact-I8 LUT M32N64 device-A/threadgroup-B shaders."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError(f"I8 LUT shader parent source drift: {before!r}")
    return text.replace(before, after)


STAGED = r'''
// Every packed G64 carries the original32-byte Q4 IDs and a16-byte signed-I8
// LUT. No coefficient quantization/reconstruction arithmetic occurs here.
template <typename Coefficient, ushort BK, ushort SG>
inline void prefill_i8_lut_stage(device const uchar *ids,
    device const int8_t *lookup, uint rank, uint first_column, uint k_origin,
    uint columns, uint contraction, uint tid, device uint *diag,
    threadgroup Coefficient *destination) {
  for (uint i=tid; i<uint(64)*BK/8; i+=uint(SG)*32) {
    const uint n=first_column+i/(BK/8), k=k_origin+(i%(BK/8))*8;
    uint packed=0;
    if (k<contraction)
      packed=*reinterpret_cast<device const uint *>(ids+
          (ulong(rank)*columns+n)*(contraction/2)+k/2);
    const ulong table=((ulong(rank)*columns+n)*(contraction/64)+k/64)*16;
#pragma unroll
    for (ushort j=0; j<8; ++j) {
      int8_t code=0;
      if (k+j<contraction) {
        code=lookup[table+((packed>>(j*4))&15u)];
        if (code==int8_t(-128)) { flash_mpp_error(diag,4u); code=0; }
      }
      destination[i*8+j]=Coefficient(code);
    }
  }
}

// Matched uncompressed comparator: same stage shape/type, descriptor and Kloop,
// with original savedI8 bytes instead of ID/LUT lookups. Source reads are bounded
// identically, including K256's final K128 down block.
template <typename Coefficient, ushort BK, ushort SG>
inline void prefill_i8_lut_stage_uncompressed(device const uchar *weights,
    uint rank,uint first_column,uint k_origin,uint columns,uint contraction,
    uint tid,device uint *diag,threadgroup Coefficient *destination) {
  for (uint i=tid; i<uint(64)*BK/8; i+=uint(SG)*32) {
    const uint n=first_column+i/(BK/8),k=k_origin+(i%(BK/8))*8;
    ulong packed=0;
    if (k<contraction)
      packed=*reinterpret_cast<device const ulong *>(weights+
          (ulong(rank)*columns+n)*contraction+k);
#pragma unroll
    for (ushort j=0; j<8; ++j) {
      int8_t code=0;
      if (k+j<contraction) {
        code=int8_t((packed>>(j*8))&255ul);
        if (code==int8_t(-128)) { flash_mpp_error(diag,4u); code=0; }
      }
      destination[i*8+j]=Coefficient(code);
    }
  }
}

template <typename Coefficient, ushort BK, ushort SG, bool Audit, bool Compressed>
inline void prefill_i8_lut_gate(device bfloat *input,device const uchar *gate,
    device const float *gate_scale,device const uchar *up,device const float *up_scale,
    device const uint *ranks,device const uint *offsets,
    device const FlashMoEBucketJob *jobs,device const uint *job_count,
    device bfloat *output,device uint *diag,constant FlashInt8ExpertStoreParams &p,
    device const int8_t *gate_lookup,device const int8_t *up_lookup,
    device float *raw_gate,device float *raw_up,device float *scaled_gate,
    device float *scaled_up,device bfloat *gate_boundary,device bfloat *up_boundary,
    uint3 group,uint3 threads,uint tid,threadgroup Coefficient *staged) {
  if (group.x>=10) { if (!tid) flash_mpp_error(diag,2u); return; }
  uint rank,begin,valid_rows;
  if (!prefill_i8_lut_shared_job<32,SG>(p,ranks,offsets,jobs,job_count,diag,
      group,threads,tid,rank,begin,valid_rows)) return;
  const uint column=group.x*64;
  constexpr auto descriptor=matmul2d_descriptor(32,64,BK,false,true,false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor,execution_simdgroups<SG>> operation;
  auto left=tensor(input+ulong(begin)*2560,
      dextents<int,2>{2560,int(valid_rows)},array<int,2>{1,2560});
  auto right=tensor(staged,extents<int,BK,64>{},array<int,2>{1,BK});
  auto gd=operation.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
  auto ud=operation.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
#pragma unroll
  for (ushort i=0; i<gd.get_capacity(); ++i)
    if (gd.is_valid_element(i)) { gd[i]=0.0f; ud[i]=0.0f; }
  for (uint k=0; k<2560; k+=BK) {
    auto a=tensor(input+ulong(begin)*2560+k,
        dextents<int,2>{BK,int(valid_rows)},array<int,2>{1,2560});
    if constexpr (Compressed)
      prefill_i8_lut_stage<Coefficient,BK,SG>(gate,gate_lookup,rank,column,k,640,2560,tid,diag,staged);
    else prefill_i8_lut_stage_uncompressed<Coefficient,BK,SG>(gate,rank,column,k,640,2560,tid,diag,staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(a,right,gd);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (Compressed)
      prefill_i8_lut_stage<Coefficient,BK,SG>(up,up_lookup,rank,column,k,640,2560,tid,diag,staged);
    else prefill_i8_lut_stage_uncompressed<Coefficient,BK,SG>(up,rank,column,k,640,2560,tid,diag,staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(a,right,ud);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i=0; i<gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index=gd.get_multidimensional_index(i);
    if (uint(index[1])>=valid_rows) continue;
    const uint n=column+index[0];
    const ulong destination=ulong(begin+index[1])*640+n;
    const float gs=gate_scale[ulong(rank)*640+n],us=up_scale[ulong(rank)*640+n];
    const float gf=gd[i]*gs,uf=ud[i]*us;
    const bfloat gv=bfloat(gf),uv=bfloat(uf);
    const bfloat silu=gv*prefill_i8_lut_shared_sigmoid(gv);
    const bfloat value=silu*uv;
    if (!(gs>0.0f) || !(us>0.0f) || !flash_mpp_finite(gs) || !flash_mpp_finite(us) ||
        !flash_mpp_finite(gf) || !flash_mpp_finite(uf) || !flash_mpp_finite(value)) flash_mpp_error(diag,4u);
    if constexpr (Audit) {
      raw_gate[destination]=gd[i]; raw_up[destination]=ud[i];
      scaled_gate[destination]=gf; scaled_up[destination]=uf;
      gate_boundary[destination]=gv; up_boundary[destination]=uv;
    } else output[destination]=value;
  }
}

template <typename Coefficient, ushort BK, ushort SG, bool Audit, bool Compressed>
inline void prefill_i8_lut_down(device bfloat *input,device const uchar *weights,
    device const float *scales,device const uint *ranks,device const uint *offsets,
    device const FlashMoEBucketJob *jobs,device const uint *job_count,
    device const uint *route_map,device bfloat *output,device uint *diag,
    constant FlashInt8ExpertStoreParams &p,device const int8_t *lookup,
    device float *raw,device float *scaled,device bfloat *boundary,
    uint3 group,uint3 threads,uint tid,threadgroup Coefficient *staged) {
  if (group.x>=40) { if (!tid) flash_mpp_error(diag,2u); return; }
  uint rank,begin,valid_rows;
  if (!prefill_i8_lut_shared_job<32,SG>(p,ranks,offsets,jobs,job_count,diag,
      group,threads,tid,rank,begin,valid_rows)) return;
  const uint column=group.x*64;
  constexpr auto descriptor=matmul2d_descriptor(32,64,BK,false,true,false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor,execution_simdgroups<SG>> operation;
  auto left=tensor(input+ulong(begin)*640,
      dextents<int,2>{640,int(valid_rows)},array<int,2>{1,640});
  auto right=tensor(staged,extents<int,BK,64>{},array<int,2>{1,BK});
  auto dot=operation.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
#pragma unroll
  for (ushort i=0; i<dot.get_capacity(); ++i)
    if (dot.is_valid_element(i)) dot[i]=0.0f;
  for (uint k=0; k<640; k+=BK) {
    auto a=tensor(input+ulong(begin)*640+k,
        dextents<int,2>{int(min(uint(BK),640u-k)),int(valid_rows)},array<int,2>{1,640});
    if constexpr (Compressed)
      prefill_i8_lut_stage<Coefficient,BK,SG>(weights,lookup,rank,column,k,2560,640,tid,diag,staged);
    else prefill_i8_lut_stage_uncompressed<Coefficient,BK,SG>(weights,rank,column,k,2560,640,tid,diag,staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(a,right,dot);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i=0; i<dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index=dot.get_multidimensional_index(i);
    if (uint(index[1])>=valid_rows) continue;
    const uint route=route_map[begin+index[1]],n=column+index[0];
    if (route>=p.route_capacity) { flash_mpp_error(diag,1u); continue; }
    const ulong destination=ulong(route)*2560+n;
    const float scale=scales[ulong(rank)*2560+n],value=dot[i]*scale;
    const bfloat rounded=bfloat(value);
    if (!(scale>0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(value) ||
        !flash_mpp_finite(rounded)) flash_mpp_error(diag,4u);
    if constexpr (Audit) {
      raw[destination]=dot[i]; scaled[destination]=value; boundary[destination]=rounded;
    } else output[destination]=rounded;
  }
}
'''


def entry(name, phase, coefficient, bk, sg, audit, control=False, compressed=True):
    is_gate = phase == "gate_up"
    args = ["device bfloat *a [[buffer(0)]]"]
    if is_gate:
        args += [f"device {'int8_t' if control else 'const uchar'} *g [[buffer(1)]]",
                 "device const float *gs [[buffer(2)]]",
                 f"device {'int8_t' if control else 'const uchar'} *u [[buffer(3)]]",
                 "device const float *us [[buffer(4)]]", "device const uint *ranks [[buffer(5)]]",
                 "device const uint *offsets [[buffer(6)]]", "device const FlashMoEBucketJob *jobs [[buffer(7)]]",
                 "device const uint *count [[buffer(8)]]",
                 f"device {'float' if audit else 'bfloat'} *out [[buffer(9)]]",
                 "device uint *diag [[buffer(10)]]", "constant FlashInt8ExpertStoreParams &p [[buffer(11)]]"]
        if not control or audit:
            args += ["device const int8_t *gl [[buffer(12)]]", "device const int8_t *ul [[buffer(13)]]"]
        if audit:
            args += ["device float *ru [[buffer(14)]]", "device float *sgate [[buffer(15)]]",
                     "device float *sup [[buffer(16)]]", "device bfloat *bg [[buffer(17)]]",
                     "device bfloat *bu [[buffer(18)]]"]
        base = "a,g,gs,u,us,ranks,offsets,jobs,count,"
        outputs = "reinterpret_cast<device bfloat *>(out),diag,p," if audit else "out,diag,p,"
        audits = "out,ru,sgate,sup,bg,bu," if audit else "nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,"
    else:
        args += [f"device {'int8_t' if control else 'const uchar'} *w [[buffer(1)]]",
                 "device const float *s [[buffer(2)]]", "device const uint *ranks [[buffer(3)]]",
                 "device const uint *offsets [[buffer(4)]]", "device const FlashMoEBucketJob *jobs [[buffer(5)]]",
                 "device const uint *count [[buffer(6)]]", "device const uint *map [[buffer(7)]]",
                 f"device {'float' if audit else 'bfloat'} *out [[buffer(8)]]", "device uint *diag [[buffer(9)]]",
                 "constant FlashInt8ExpertStoreParams &p [[buffer(10)]]"]
        if not control or audit:
            args += ["device const int8_t *lut [[buffer(11)]]"]
        if audit:
            args += ["device float *scaled [[buffer(12)]]", "device bfloat *boundary [[buffer(13)]]"]
        base = "a,w,s,ranks,offsets,jobs,count,map,"
        outputs = "reinterpret_cast<device bfloat *>(out),diag,p," if audit else "out,diag,p,"
        audits = "out,scaled,boundary," if audit else "nullptr,nullptr,nullptr,"
    args += ["uint3 group [[threadgroup_position_in_grid]]", "uint3 threads [[threads_per_threadgroup]]",
             "uint tid [[thread_index_in_threadgroup]]"]
    if control:
        function = f"prefill_i8_lut_control_{'gate' if is_gate else 'down'}<32,{sg},{bk},{'true' if bk else 'false'},{'true' if audit else 'false'}>"
        call = function + "(" + base + outputs + audits + "group,threads,tid);"
        body = ("  (void)gl; (void)ul;\n" if is_gate else "  (void)lut;\n") if audit else ""
    else:
        function = f"prefill_i8_lut_{'gate' if is_gate else 'down'}<{coefficient},{bk},{sg},{'true' if audit else 'false'},{'true' if compressed else 'false'}>"
        call = function + "(" + base + outputs + ("gl,ul," if is_gate else "lut,") + audits + "group,threads,tid,staged);"
        body = f"  alignas(16) threadgroup {coefficient} staged[64*{bk}];\n"
    return f"[[max_total_threads_per_threadgroup({sg*32})]] kernel void {name}(\n    " + ",\n    ".join(args) + ") {\n" + body + "  " + call + "\n}\n"


def main():
    parent = (ROOT / "dev/benchmarks/prefill_moe_sep21/memory.metal").read_text()
    parent = parent[:parent.index("#define PREFILL4K_INT8TILES_GATE")]
    parent = parent.replace("prefill_moe_sep21_memory_", "prefill_i8_lut_shared_")
    boundary = parent.index("template <ushort M, ushort SG, ushort K = 0, bool Static = false>\ninline void prefill_i8_lut_shared_gate")
    prefix, controls = parent[:boundary], parent[boundary:]
    # This bounded representation is Full512-only: an absent rank is malformed,
    # rather than an eligible Q4 miss as in the general store implementation.
    prefix = replace(prefix, "  if (rank == UINT_MAX) return false;",
                     "  if (rank == UINT_MAX) { if (!tid) flash_mpp_error(diag,1u); return false; }")
    controls = controls.replace("prefill_i8_lut_shared_gate", "prefill_i8_lut_control_gate").replace(
        "prefill_i8_lut_shared_down", "prefill_i8_lut_control_down")
    controls = replace(controls, "bool Static = false>\ninline void", "bool Static = false, bool Audit = false>\ninline void", 2)
    controls = replace(controls, "constant FlashInt8ExpertStoreParams &p,\n    uint3 group", "constant FlashInt8ExpertStoreParams &p,\n    device float *raw_gate, device float *raw_up, device float *scaled_gate, device float *scaled_up,\n    device bfloat *gate_boundary, device bfloat *up_boundary,\n    uint3 group")
    controls = replace(controls, "constant FlashInt8ExpertStoreParams &p, uint3 group", "constant FlashInt8ExpertStoreParams &p,\n    device float *raw, device float *scaled, device bfloat *boundary, uint3 group")
    controls = replace(controls, "    output[ulong(begin + index[1]) * 640 + n] = value;", """    const ulong destination=ulong(begin+index[1])*640+n;
    if constexpr (Audit) {
      raw_gate[destination]=gd[i]; raw_up[destination]=ud[i];
      scaled_gate[destination]=gf; scaled_up[destination]=uf;
      gate_boundary[destination]=gv; up_boundary[destination]=uv;
    } else output[destination]=value;""")
    controls = replace(controls, "    output[ulong(route) * 2560 + n] = value;", """    const ulong destination=ulong(route)*2560+n;
    if constexpr (Audit) {
      raw[destination]=dot[i]; scaled[destination]=dot[i]*scale; boundary[destination]=value;
    } else output[destination]=value;""")
    source = "// Generated by prepare.py; isolated threadgroup-B I8 LUT candidate.\n" + prefix + controls + STAGED
    for bk, sg in ((0, 4), (128, 2)):
        label = "whole_sg4" if not bk else "fixed_k128_sg2"
        for phase in ("gate_up", "down_scatter"):
            for audit in (False, True):
                name = f"prefill_i8_lut_{phase}_m32_n64_{label}" + ("_audit" if audit else "")
                source += entry(name, phase, "int8_t", bk, sg, audit, True)
    for bk in (128, 256):
        for sg in (2, 4):
            for label, coefficient in (("i8", "int8_t"), ("bf16", "bfloat")):
                for phase in ("gate_up", "down_scatter"):
                    for audit in (False, True):
                        name = f"prefill_i8_lut_{phase}_m32_n64_k{bk}_sg{sg}_{label}" + ("_audit" if audit else "")
                        source += entry(name, phase, coefficient, bk, sg, audit)
                        matched = f"prefill_i8_lut_{phase}_m32_n64_k{bk}_sg{sg}_{label}_uncompressed" + ("_audit" if audit else "")
                        source += entry(matched, phase, coefficient, bk, sg, audit, compressed=False)
    source += "#endif\n"
    (HERE / "candidate.metal").write_text(source)


if __name__ == "__main__":
    main()
