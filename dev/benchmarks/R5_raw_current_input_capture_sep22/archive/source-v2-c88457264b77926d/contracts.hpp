#pragma once
#include "flash/FlashAffine.hpp"
#include <array>
#include <string>

namespace splash::flash::r5_raw_current_capture_sep22 {
inline constexpr char kParentSource[]="4bb7b637c2b6b60520159ad3724a8d9e3867c7c538dd8d260caa4fc7d184769a";
inline constexpr char kParentExe[]="6f7e22a2ca9c0c9728bf356391d2bde17c4e9bc90ab6295e625cac15e7987d68";
inline constexpr char kLibrary[]="dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6";
inline constexpr char kShapeMetadataSha[]="80560c70fc06e700dbd55e2819dbf08c2a4bb20593275ea349bd0f3b8cfeb685";
inline constexpr char kRootTokenFile[]="/Users/mweinbach/Projects/splash/build/release/flash/sep22-fixed4-qualified-http-exact2048-Root-tokens-v1.json";
inline constexpr char kRootTokenFileSha[]="72e5a23f0504ba862d43c22e01e820fc4007b3b939ec4cbc8518aaee499832fb";
inline constexpr char kRootTokenU32LESha[]="55cf1a355b4a2c97012c752b87955198ef3bb1f1b992b3fb48d35ff7659f3795";
inline constexpr uint64_t kOwnerReservation=16ULL<<20,kHostArenaBytes=64ULL<<20,kHostHeadroomRequired=128ULL<<20;
inline constexpr uint64_t kLogicalBytes=4362240,kAlignedBytes=5046272,kSpillLimit=(4ULL<<30)-1;
inline constexpr uint32_t kRoles=113,kRequestId=1,kGeneration=1,kOrdinal=3,kPhysicalRows=5;
struct Shape final { uint32_t K,N,bits,group,calls;const char *pipeline; };
inline constexpr std::array<Shape,7> kShapes{{
 {6144,2560,5,128,36,"flash_affine_mlx_qmv_f32xsum_v1_q5_g128"},
 {2560,10240,4,64,27,"flash_affine_mlx_qmv_f32xsum_v1_q4_g64"},
 {2560,6144,5,128,26,"flash_affine_mlx_qmv_f32xsum_v1_q5_g128"},
 {2560,6144,6,64,10,"flash_affine_mlx_qmv_f32xsum_v1_q6_g64"},
 {2560,12288,4,64,5,"flash_affine_mlx_qmv_f32xsum_v1_q4_g64"},
 {2560,10240,5,64,4,"flash_affine_mlx_qmv_f32xsum_v1_q5_g64"},
 {6144,2560,4,64,5,"flash_affine_mlx_qmv_f32xsum_v1_q4_g64"}}};
struct Role final { std::string name;FlashAffineProjection projection;uint32_t shape=0; };
inline uint64_t logicalBytes(const Role &r){return uint64_t{kPhysicalRows}*r.projection.inputSize*2;}
inline uint64_t guardedBytes(const Role &r){return (logicalBytes(r)+128+16383)&~uint64_t{16383};}
}
