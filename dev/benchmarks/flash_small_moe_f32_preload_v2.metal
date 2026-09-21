// Private exact-lane small-window MoE: F32 coefficients and independent
// original contiguous-lane reductions. No BF16 coefficient conversion/MPP.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "FlashSmallMoEF32ABI.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
inline bool small_moe_finite(float x) {return (as_type<uint>(x)&0x7f800000u)!=0x7f800000u;}
inline float small_moe_nan() {return as_type<float>(0x7fc00000u);}
inline void small_moe_error(device atomic_uint *d,uint f) {atomic_fetch_or_explicit(d,f,memory_order_relaxed);}
#pragma METAL fp math_mode(fast)
inline bfloat small_moe_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent=bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator=bfloat(1.0f)+exponent;
  const bfloat tail=bfloat(1.0f)/denominator;
  return source<bfloat(0.0f)?tail:bfloat(1.0f)-tail;
}
#pragma METAL fp math_mode(safe)
inline bfloat small_moe_activation(bfloat gate,bfloat up,device atomic_uint *diag) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat sigmoid=small_moe_sigmoid(gate);
  const bfloat silu=gate*sigmoid;
  const bfloat value=silu*up;
  if(!small_moe_finite(float(gate)) || !small_moe_finite(float(up)) || !small_moe_finite(float(value))) {
    small_moe_error(diag,4u);return bfloat(small_moe_nan());
  }
  return value;
}

kernel void flash_small_moe_f32_preload_v2_jobs(device const long *ids [[buffer(0)]],
    device FlashSmallMoEF32Job *jobs [[buffer(1)]],device uint *count [[buffer(2)]],
    device atomic_uint *diag [[buffer(3)]],constant FlashSmallMoEF32JobParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]]) {
  if(group.x || group.y || group.z || threads.x!=256 || threads.y!=1 || threads.z!=1 ||
      !p.rows || p.rows>16 || p.selections!=10 || p.route_capacity!=p.rows*10 ||
      (p.group_rows!=2 && p.group_rows!=4) || p.reserved0 || p.reserved1 || p.reserved2 || p.reserved3) {
    if(!tid){count[0]=0;small_moe_error(diag,2u);}return;
  }
  if(tid<p.route_capacity) jobs[tid]=FlashSmallMoEF32Job{-1,0,{UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX},0};
  threadgroup uint totals[8],begins[8];
  uint leader=0;
  if(tid<p.route_capacity) {
    const long expert=ids[tid];uint before=0;
    for(uint r=0;r<tid;++r) before+=uint(ids[r]==expert);
    leader=uint(before%p.group_rows==0);
  }
  const uint local=simd_prefix_exclusive_sum(leader),total=simd_sum(leader);
  if(!lane)totals[sg]=total;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if(!tid){uint base=0;for(uint i=0;i<8;++i){begins[i]=base;base+=totals[i];}count[0]=base;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(leader) {
    FlashSmallMoEF32Job job{-1,0,{UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX},0};
    job.expert=ids[tid];
    for(uint r=tid;r<p.route_capacity && job.route_count<p.group_rows;++r)
      if(ids[r]==job.expert)job.routes[job.route_count++]=r;
    const uint destination=begins[sg]+local;
    if(destination>=p.route_capacity){small_moe_error(diag,2u);return;}
    jobs[destination]=job;
  }
}

template<ushort GR>
inline bool small_moe_job(device const long *ids,device const FlashSmallMoEF32Job *jobs,
    device const uint *count,uint index,uint routes,thread FlashSmallMoEF32Job &job,
    device atomic_uint *diag) {
  if constexpr(GR==1){if(index>=routes)return false;job={ids[index],1,{index,UINT_MAX,UINT_MAX,UINT_MAX},0};}
  else {
    if(count[0]>routes){small_moe_error(diag,2u);return false;}
    if(index>=count[0])return false;job=jobs[index];
    if(!job.route_count || job.route_count>GR || job.reserved){small_moe_error(diag,2u);return false;}
    for(uint i=0;i<job.route_count;++i)
      if(job.routes[i]>=routes || ids[job.routes[i]]!=job.expert || (i && job.routes[i]<=job.routes[i-1])){
        small_moe_error(diag,2u);return false;
      }
  }
  return true;
}
inline bool small_moe_projection(constant FlashAffineParams &p,uint rows,uint k,uint n,uint flags) {
  return rows && rows<=16 && p.rows==rows && p.selections==10 && p.input_size==k && p.output_size==n &&
      p.experts==512 && p.bits==4 && p.group_size==64 && p.flags==flags &&
      p.weight_row_stride_bytes>=k/2 && !(p.weight_row_stride_bytes%4) &&
      !(p.weight_expert_stride_bytes%4) && p.parameter_row_stride_bytes>=ulong(k/64)*2 &&
      !(p.parameter_row_stride_bytes%2) && !(p.parameter_expert_stride_bytes%2);
}

template<ushort GR>
inline void small_moe_gate(device const bfloat *input,device const uchar *gw,
    device const uchar *gs,device const uchar *gb,device const uchar *uw,device const uchar *us,
    device const uchar *ub,device const long *ids,device const FlashSmallMoEF32Job *jobs,
    device const uint *job_count,device bfloat *gout,device bfloat *uout,device bfloat *activation,
    device float *gtap,device float *utap,device atomic_uint *diag,constant FlashSmallMoEF32GateParams &p,
    uint3 group,uint3 threads,uint simd,uint lane) {
  if(!small_moe_projection(p.gate,p.gate.rows,2560,640,1) ||
      !small_moe_projection(p.up,p.gate.rows,2560,640,1) || p.group_rows!=GR ||
      p.job_capacity!=p.gate.rows*10 || p.write_f32_taps>1 || p.reserved ||
      threads.x!=128 || threads.y!=1 || threads.z!=1 || group.x>=80 || group.z) {
    if(!lane)small_moe_error(diag,2u);return;
  }
  FlashSmallMoEF32Job job;
  if(!small_moe_job<GR>(ids,jobs,job_count,group.y,p.job_capacity,job,diag))return;
  const uint nbase=group.x*8+simd*2;
  if(job.expert<0 || ulong(job.expert)>=512) {
    if(!lane){small_moe_error(diag,1u|4u);for(uint r=0;r<job.route_count;++r)for(ushort c=0;c<2;++c){
      const ulong out=ulong(job.routes[r])*640+nbase+c;
      gout[out]=uout[out]=activation[out]=bfloat(small_moe_nan());
      if(p.write_f32_taps){gtap[out]=utap[out]=small_moe_nan();}
    }}return;
  }
  float gsum[GR][2],usum[GR][2];
#pragma unroll
  for(ushort r=0;r<GR;++r) {
#pragma unroll
    for(ushort c=0;c<2;++c){gsum[r][c]=0.0f;usum[r][c]=0.0f;}
  }
  for(uint block=0;block<2560;block+=16*32) {
    const uint origin=block+lane*16;
    if(origin>=2560)continue;
    float xvalues[GR][16];
#pragma unroll
    for(ushort r=0;r<GR;++r) {
#pragma unroll
      for(ushort j=0;j<16;++j)
        xvalues[r][j]=r<job.route_count?float(input[ulong(job.routes[r]/10)*2560+origin+j]):0.0f;
    }
#pragma unroll
    for(ushort c=0;c<2;++c) {
      const ulong gn=ulong(job.expert)*p.gate.weight_expert_stride_bytes+ulong(nbase+c)*p.gate.weight_row_stride_bytes;
      const ulong un=ulong(job.expert)*p.up.weight_expert_stride_bytes+ulong(nbase+c)*p.up.weight_row_stride_bytes;
      const ulong gp=ulong(job.expert)*p.gate.parameter_expert_stride_bytes+ulong(nbase+c)*p.gate.parameter_row_stride_bytes+ulong(origin/64)*2;
      const ulong up=ulong(job.expert)*p.up.parameter_expert_stride_bytes+ulong(nbase+c)*p.up.parameter_row_stride_bytes+ulong(origin/64)*2;
      const float gsf=float(*reinterpret_cast<device const bfloat *>(gs+gp)),gbi=float(*reinterpret_cast<device const bfloat *>(gb+gp));
      const float usf=float(*reinterpret_cast<device const bfloat *>(us+up)),ubi=float(*reinterpret_cast<device const bfloat *>(ub+up));
      const device uint *gwords=reinterpret_cast<device const uint *>(gw+gn),*uwords=reinterpret_cast<device const uint *>(uw+un);
#pragma unroll
      for(ushort pack=0;pack<2;++pack) {
        const uint gcodes=gwords[origin/8+pack],ucodes=uwords[origin/8+pack];
#pragma unroll
        for(ushort j=0;j<8;++j) {
          const float gweight=float((gcodes>>(j*4))&15u)*gsf+gbi;
          const float uweight=float((ucodes>>(j*4))&15u)*usf+ubi;
#pragma unroll
          for(ushort r=0;r<GR;++r)if(r<job.route_count) {
            const float x=xvalues[r][pack*8+j];
            gsum[r][c]+=x*gweight;usum[r][c]+=x*uweight;
          }
        }
      }
    }
  }
#pragma unroll
  for(ushort c=0;c<2;++c) {
#pragma unroll
    for(ushort r=0;r<GR;++r) {
    if(r>=job.route_count)continue;
    const float g=simd_sum(gsum[r][c]),u=simd_sum(usum[r][c]);
    if(!lane && r<job.route_count){const ulong out=ulong(job.routes[r])*640+nbase+c;
      const bfloat gv=bfloat(g),uv=bfloat(u);
      if(!small_moe_finite(g) || !small_moe_finite(u) || !small_moe_finite(float(gv)) || !small_moe_finite(float(uv)))small_moe_error(diag,4u);
      gout[out]=gv;uout[out]=uv;activation[out]=small_moe_activation(gv,uv,diag);
      if(p.write_f32_taps){gtap[out]=g;utap[out]=u;}
    }
  }
  }
}

template<ushort GR>
inline void small_moe_down(device const bfloat *input,device const uchar *w,device const uchar *s,
    device const uchar *b,device const long *ids,device const FlashSmallMoEF32Job *jobs,device const uint *job_count,
    device bfloat *output,device float *tap,device atomic_uint *diag,constant FlashSmallMoEF32DownParams &params,
    uint3 group,uint3 threads,uint simd,uint lane) {
  const auto &p=params.affine;
  if(!small_moe_projection(p,p.rows,640,2560,3) || params.group_rows!=GR || params.job_capacity!=p.rows*10 ||
      params.write_f32_taps>1 || params.reserved || threads.x!=64 || threads.y!=1 || threads.z!=1 || group.x>=320 || group.z){
    if(!lane)small_moe_error(diag,2u);return;
  }
  FlashSmallMoEF32Job job;
  if(!small_moe_job<GR>(ids,jobs,job_count,group.y,params.job_capacity,job,diag))return;
  const uint nbase=group.x*8+simd*4;
  if(job.expert<0 || ulong(job.expert)>=512) {
    if(!lane){small_moe_error(diag,1u);for(uint r=0;r<job.route_count;++r)for(ushort c=0;c<4;++c){
      const ulong out=ulong(job.routes[r])*2560+nbase+c;output[out]=bfloat(small_moe_nan());
      if(params.write_f32_taps)tap[out]=small_moe_nan();
    }}return;
  }
  float sum[GR][4];
#pragma unroll
  for(ushort r=0;r<GR;++r) {
#pragma unroll
    for(ushort c=0;c<4;++c)sum[r][c]=0.0f;
  }
  for(uint block=0;block<640;block+=8*32){const uint origin=block+lane*8;if(origin>=640)continue;
    float xvalues[GR][8];
#pragma unroll
    for(ushort r=0;r<GR;++r) {
#pragma unroll
      for(ushort j=0;j<8;++j)
        xvalues[r][j]=r<job.route_count?float(input[ulong(job.routes[r])*640+origin+j]):0.0f;
    }
#pragma unroll
    for(ushort c=0;c<4;++c){
      const ulong wn=ulong(job.expert)*p.weight_expert_stride_bytes+ulong(nbase+c)*p.weight_row_stride_bytes;
      const ulong pn=ulong(job.expert)*p.parameter_expert_stride_bytes+ulong(nbase+c)*p.parameter_row_stride_bytes+ulong(origin/64)*2;
      const float sf=float(*reinterpret_cast<device const bfloat *>(s+pn)),bias=float(*reinterpret_cast<device const bfloat *>(b+pn));
      const uint codes=reinterpret_cast<device const uint *>(w+wn)[origin/8];
#pragma unroll
      for(ushort j=0;j<8;++j){const float weight=float((codes>>(j*4))&15u)*sf+bias;
#pragma unroll
        for(ushort r=0;r<GR;++r)if(r<job.route_count){
          const float x=xvalues[r][j];sum[r][c]+=x*weight;
        }
      }
    }
  }
#pragma unroll
  for(ushort c=0;c<4;++c) {
#pragma unroll
    for(ushort r=0;r<GR;++r){if(r>=job.route_count)continue;const float value=simd_sum(sum[r][c]);
    if(!lane && r<job.route_count){const ulong out=ulong(job.routes[r])*2560+nbase+c;
      const bfloat result=bfloat(value);if(!small_moe_finite(value) || !small_moe_finite(float(result)))small_moe_error(diag,4u);
      output[out]=result;if(params.write_f32_taps)tap[out]=value;
    }
  }
  }
}

#define SMALL_GATE(GR) \
kernel void flash_small_moe_f32_preload_v2_gate_up_gr##GR(device const bfloat *a [[buffer(0)]],device const uchar *gw [[buffer(1)]], \
    device const uchar *gs [[buffer(2)]],device const uchar *gb [[buffer(3)]],device const uchar *uw [[buffer(4)]], \
    device const uchar *us [[buffer(5)]],device const uchar *ub [[buffer(6)]],device const long *ids [[buffer(7)]], \
    device const FlashSmallMoEF32Job *jobs [[buffer(8)]],device const uint *count [[buffer(9)]], \
    device bfloat *g [[buffer(10)]],device bfloat *u [[buffer(11)]],device bfloat *act [[buffer(12)]], \
    device float *gt [[buffer(13)]],device float *ut [[buffer(14)]],device atomic_uint *d [[buffer(15)]], \
    constant FlashSmallMoEF32GateParams &p [[buffer(16)]],uint3 grp [[threadgroup_position_in_grid]], \
    uint3 th [[threads_per_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
  small_moe_gate<GR>(a,gw,gs,gb,uw,us,ub,ids,jobs,count,g,u,act,gt,ut,d,p,grp,th,sg,lane); \
}
#define SMALL_DOWN(GR) \
kernel void flash_small_moe_f32_preload_v2_down_gr##GR(device const bfloat *a [[buffer(0)]],device const uchar *w [[buffer(1)]], \
    device const uchar *s [[buffer(2)]],device const uchar *b [[buffer(3)]],device const long *ids [[buffer(4)]], \
    device const FlashSmallMoEF32Job *jobs [[buffer(5)]],device const uint *count [[buffer(6)]], \
    device bfloat *out [[buffer(7)]],device float *tap [[buffer(8)]],device atomic_uint *d [[buffer(9)]], \
    constant FlashSmallMoEF32DownParams &p [[buffer(10)]],uint3 grp [[threadgroup_position_in_grid]], \
    uint3 th [[threads_per_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
  small_moe_down<GR>(a,w,s,b,ids,jobs,count,out,tap,d,p,grp,th,sg,lane); \
}
SMALL_GATE(2)
SMALL_DOWN(2)
#undef SMALL_GATE
#undef SMALL_DOWN
