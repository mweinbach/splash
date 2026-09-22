#pragma once
#include <cstdint>

namespace splash::flash::raw_large_r4_trained_state_oracle_sep22 {
inline constexpr char kControlBuild[]="build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2";
inline constexpr char kControlSource[]="162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a";
inline constexpr char kControlExe[]="663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438";
inline constexpr char kControlLibrary[]="7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8";
inline constexpr char kCandidateFlag[]="SPLASH_FLASH_RAW_LARGE_R4_GUARD_PAIR_SEP22";
inline constexpr char kProofFlag[]="SPLASH_FLASH_DIAG_RAW_LARGE_R4_TRAINED_STATE_SEP22";
inline constexpr char kRootTokenFile[]="/Users/mweinbach/Projects/splash/build/release/flash/sep22-fixed4-qualified-http-exact2048-Root-tokens-v1.json";
inline constexpr char kRootTokenFileSha[]="72e5a23f0504ba862d43c22e01e820fc4007b3b939ec4cbc8518aaee499832fb";
inline constexpr char kRootTokenU32LESha[]="55cf1a355b4a2c97012c752b87955198ef3bb1f1b992b3fb48d35ff7659f3795";
inline constexpr uint32_t kRequestId=1,kGeneration=1,kDraftDepth=3,kPhysicalRows=4,kSelectedOrdinal=3;
inline constexpr uint32_t kNativeCapacity=16384,kPromptRows=2048,kOutputTokens=64;
inline constexpr uint32_t kMainStatePlanes=134,kInitializedLazyPlanes=216,kDefinedPLEPlanes=2,kTrainedHeadPlanes=5;
inline constexpr uint32_t kLegacyQ4CallsPerR4=26,kNewPairCallsPerR4=87,kTotalLargeRawCallsPerR4=113;
inline constexpr uint64_t kSpillLimit=(4ULL<<30)-1,kHostPreallocation=64ULL<<20,kHostHeadroom=128ULL<<20;
// Recorder admission only; it must allocate NO new backend-owned buffers.
inline constexpr uint64_t kRecorderReservation=16ULL<<20,kNewBackendOwnedBytes=0,kInputCopyDispatches=0;
}
