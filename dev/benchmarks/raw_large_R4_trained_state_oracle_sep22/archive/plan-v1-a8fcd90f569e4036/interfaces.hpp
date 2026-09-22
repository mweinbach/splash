#pragma once
#include "contracts.hpp"
#include "flash/FlashForward.hpp"
#include "flash/FlashMTP.hpp"
#include "engine/MemoryGovernor.hpp"
#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::flash::raw_large_r4_trained_state_oracle_sep22 {
// Source-plan interfaces only. Implementations follow Root scope review and
// sealed candidate READY. No producer hook or raw-input capture is declared.
enum class Phase:uint32_t { ThirdPendingVerify4,ThirdResolvedCommit,NextActualTarget };
enum class Type:uint32_t { BF16=1,F32=2,I64=3,U32=4 };
struct Plane final {std::string label;Type type;metal::MetalBuffer view;uint64_t exportedBytes=0;bool fullyInitialized=false;};
struct Cookie final {uint64_t request=1,generation=1,targetOrdinal=0,targetBegin=0;uint32_t depth=3,physicalRows=4;};
struct WorkerCarry final {
 uint64_t targetBegin=0,foldedHeadOffset=0,currentHeadOffset=0,emittedBeforeCycle=0;
 uint32_t retained=0;
 std::optional<uint32_t> pendingToken;
 std::span<const uint32_t> realVerificationTokens,committedFoldTokens;
 metal::MetalBuffer ownedFoldHiddenBF16;
};
struct Binding final {
 std::array<uintptr_t,3> mainImplOwnerIdentity{};
 std::array<uintptr_t,2> headImplOwner{};
 std::vector<metal::MetalBuffer> mainPhysical,headPhysical,knownLazyPhysical;
};
struct CampaignPlan final {
 uint32_t capacity=0;
 uint64_t mainPerFrame=0,headPerFrame=0,lazyAndDefinedPLEPerFrame=0;
 uint64_t localUnknownTailsReservePerFrame=0,workerCarryReservePerFrame=0;
 uint64_t targetOutputPerWindow=0,metadataAndFileHeaders=0,total=0;
};
struct PolicyBinding final {
 std::string measuredCommandSha256,canonicalAllFlagsJson,trajectorySha256;
 std::string controlCodeSource,candidateCodeSource,controlExe,candidateExe,controlLibrary,candidateLibrary;
 // Same winner-pref flags on both arms: FMA1/HCnorm1/W8=0/QSA2pass=0.
 // Actual binding MUST include every other measured flag and fixed depth3.
};
class Recorder;
class TypedTargetScope final {
public:
 TypedTargetScope(Recorder&,Cookie);
 ~TypedTargetScope();
 TypedTargetScope(const TypedTargetScope&)=delete;
 TypedTargetScope&operator=(const TypedTargetScope&)=delete;
 void completeSuccessfulSynchronousTarget();
};
class Recorder final {
public:
 // BEFORE host allocation, scope, state/tensor payload read or export.
 static CampaignPlan sourcePlan(uint32_t capacity);
 Recorder(metal::MetalBackend&,engine::MemoryGovernor&,FlashForward&,FlashMTPForward&,CampaignPlan,PolicyBinding);
 Recorder(const Recorder&)=delete;Recorder&operator=(const Recorder&)=delete;
 // Metadata/extent-only actual crosscheck of ALL three frames before first
 // payload read/export; constructor verifies no extra backend owners.
 void actualPreflight(const FlashRequestState&,const FlashMTPState&,const FlashForwardResult&,const WorkerCarry&);
 // Both persistent state owners must be healthy at completed command bounds.
 void pending(const Cookie&,const FlashRequestState&,const FlashMTPState&,const FlashForwardResult&,const WorkerCarry&);
 // AFTER original commitVerify/head.truncate/copyHidden/fold-token assignment
 // and existing exact offset assertions. Target state is now resolved.
 void resolved(const Cookie&,const FlashRequestState&,const FlashMTPState&,const WorkerCarry&);
 void future(const Cookie&,const FlashRequestState&,const FlashMTPState&,const FlashForwardResult&,const WorkerCarry&);
 // Releases diagnostic reservation/host storage; final zero reservation and
 // zero extra owners are checked before clean whole-process teardown.
 void finish();
};
}
