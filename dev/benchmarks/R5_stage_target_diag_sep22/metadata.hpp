#pragma once
#include "metal/abi/FlashAffine.h"
#include "metal/abi/FlashFloatDenseCache.h"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashInt8Head.h"
#include "metal/abi/FlashMoEBuckets.h"
#include <cstring>
#include <sstream>
namespace splash::flash::r5_stage_target_diag_sep22 {
template<class T,class D>bool params(const D&d,T&out){if(d.bytes.size()!=1||d.bytes[0].sizeBytes!=sizeof(T)||!d.bytes[0].data)return false;std::memcpy(&out,d.bytes[0].data,sizeof(T));return true;}
template<class D>std::string decodeParameters(const D&d){
 std::ostringstream o;const std::string_view n=d.pipelineName;
 if(n.starts_with("flash_float_dense_small_rows")||n.starts_with("flash_qsa_out_f32")){
  FlashFloatDenseSmallRowsParams p{};if(!params(d,p))return "{\"ABI\":\"floatDense\",\"decoded\":false}";
  o<<"{\"ABI\":\"FlashFloatDenseSmallRowsParams\",\"decoded\":true,\"rows\":"<<p.rows<<",\"padded_rows\":"<<p.padded_rows<<",\"K\":"<<p.input_size<<",\"N\":"<<p.output_size<<",\"output_begin\":"<<p.output_begin<<",\"output_count\":"<<p.output_count<<",\"M\":"<<p.tile_rows<<",\"tile_N\":"<<p.tile_outputs<<",\"coefficient_dtype\":\"F32\",\"input_dtype\":\"BF16\"}";
 }else if(n.starts_with("flash_affine_qmv")||n.starts_with("flash_affine_scalar")||n.starts_with("flash_affine_b")){
  FlashAffineParams p{};if(!params(d,p))return "{\"ABI\":\"affine\",\"decoded\":false}";
  o<<"{\"ABI\":\"FlashAffineParams\",\"decoded\":true,\"rows\":"<<p.rows<<",\"selections\":"<<p.selections<<",\"K\":"<<p.input_size<<",\"N\":"<<p.output_size<<",\"experts\":"<<p.experts<<",\"bits\":"<<p.bits<<",\"group_size\":"<<p.group_size<<",\"flags\":"<<p.flags<<",\"weight_row_stride_bytes\":"<<p.weight_row_stride_bytes<<",\"parameter_row_stride_bytes\":"<<p.parameter_row_stride_bytes<<"}";
 }else if(n=="expert_r5_compact_native_sep22_plan"||n=="flash_moe_direct_a_pack"||n.starts_with("flash_moe_bucket")){
  FlashMoEBucketParams p{};if(!params(d,p))return "{\"ABI\":\"bucket\",\"decoded\":false}";
  o<<"{\"ABI\":\"FlashMoEBucketParams\",\"decoded\":true,\"rows\":"<<p.rows<<",\"selections\":"<<p.selections<<",\"width\":"<<p.width<<",\"experts\":"<<p.experts<<",\"routes\":"<<p.routes<<",\"M\":"<<p.tile_rows<<",\"job_capacity\":"<<p.job_capacity<<"}";
 }else if(n.starts_with("flash_int8_expert_store_gate_up_m")||n.starts_with("flash_int8_expert_store_down_scatter_m")){
  FlashInt8ExpertStoreParams p{};if(!params(d,p))return "{\"ABI\":\"int8Expert\",\"decoded\":false}";
  o<<"{\"ABI\":\"FlashInt8ExpertStoreParams\",\"decoded\":true,\"rows\":"<<p.rows<<",\"selections\":"<<p.selections<<",\"routes\":"<<p.route_capacity<<",\"M\":"<<p.tile_rows<<",\"stored_experts\":"<<p.stored_experts<<",\"job_capacity\":"<<p.job_capacity<<",\"coefficient_dtype\":\"I8\"}";
 }else if(n.starts_with("flash_int8_head")){
  FlashInt8HeadParams p{};if(!params(d,p))return "{\"ABI\":\"int8Head\",\"decoded\":false}";
  o<<"{\"ABI\":\"FlashInt8HeadParams\",\"decoded\":true,\"rows\":"<<p.rows<<",\"padded_rows\":"<<p.padded_rows<<",\"K\":"<<p.input_size<<",\"N\":"<<p.output_size<<",\"bits\":"<<p.bits<<",\"group_size\":"<<p.group_size<<",\"M\":"<<p.tile_rows<<",\"tile_N\":"<<p.tile_outputs<<"}";
 }else{return "null";}
 return o.str();
}
}
