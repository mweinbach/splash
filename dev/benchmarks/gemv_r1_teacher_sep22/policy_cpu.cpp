// CPU-only qualification of the strict Teacher V5 standard-decode scope.
// Headers contain no backend, model, device, payload or execution workspace.
#include "scope.hpp"
#include "dev/benchmarks/gemv_decode_r1_worker_sep21/bridge.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace vector = splash::flash::gemv_decode_r1_sep21;
namespace scope = splash::flash::gemv_r1_teacher_sep22;

namespace {
constexpr std::string_view kSourceID =
    "bd384554f00dafdc285bf6df2dd34bebfe2d963fa1c955e0834c8a08e656f29b";
constexpr std::string_view kScopeSchema =
    ";scope=strict-standard-singleton-R1-no-mtpState-no-prefill-no-verify-no-head-no-batch-v1";
uint64_t checks = 0;

void require(bool passed, const char *message) {
  ++checks;
  if (!passed) throw std::runtime_error(message);
}
template <class F> void requiresInvalid(F &&operation, const char *message) {
  bool caught = false;
  try { operation(); } catch (const std::invalid_argument &) { caught = true; }
  require(caught, message);
}
template <class F> void requiresChangedState(F &&operation, const char *message) {
  bool caught = false;
  try { operation(); }
  catch (const std::invalid_argument &) {
    throw std::runtime_error("A valid optional-state mutation was classified as invalid syntax");
  } catch (const std::logic_error &) { caught = true; }
  require(caught, message);
}
void setFlag(const char *value) {
  require((value ? setenv(vector::kFlag, value, 1) : unsetenv(vector::kFlag)) == 0,
      "Cannot set CPU-only optional-state fixture environment");
}
std::string_view stateName(vector::SwitchState state) {
  switch (state) {
    case vector::SwitchState::Missing: return "missing";
    case vector::SwitchState::Disabled: return "0";
    case vector::SwitchState::Enabled: return "1";
  }
  throw std::runtime_error("Unexpected parsed optional policy state");
}

void parserChecks() {
  require(vector::parseState(nullptr) == vector::SwitchState::Missing, "Missing optional state was collapsed");
  require(vector::parseState("0") == vector::SwitchState::Disabled, "Literal zero optional state differs");
  require(vector::parseState("1") == vector::SwitchState::Enabled, "Literal one optional state differs");
  require(!vector::parseSwitch(nullptr) && !vector::parseSwitch("0") && vector::parseSwitch("1"),
      "Missing/0/1 parser enablement differs");
  for (const char *bad : {"", " ", " 0", "0 ", " 1", "1 ", "00", "01", "10", "2", "-0", "-1", "+1",
       "true", "false", "TRUE", "False", "yes", "no", "on", "off", "\t1", "1\n", "0\r", "1.0", "0x1"}) {
    requiresInvalid([&] { (void)vector::parseState(bad); }, "Malformed optional policy state was accepted");
    requiresInvalid([&] { (void)vector::parseSwitch(bad); }, "Malformed truthy/whitespace policy was accepted");
  }
}

void identityChecks() {
  static_assert(vector::sourceIdentityValid(vector::kSourceIdentitySha256));
  static_assert(std::is_trivially_copyable_v<vector::Counters>);
  require(std::string_view(vector::kSourceIdentitySha256) == kSourceID,
      "Inherited vector source/certificate identity changed");
  require(scope::kScopeSchema == kScopeSchema, "Strict standard-only scope schema differs");
  for (const std::string &bad : std::array<std::string, 6>{"", std::string(63, '0'), std::string(65, '0'),
       std::string(64, 'G'), std::string(64, 'A'), std::string(31, '0') + '\0' + std::string(32, '0')})
    require(!vector::sourceIdentityValid(bad), "Malformed source identity was accepted");
  require(vector::sourceIdentityValid(std::string(64, 'a')), "Valid lowercase SHA256 syntax was rejected");
  require(vector::markerFor(false).empty() && scope::markerFor(false).empty(),
      "Disabled policy changed immutable baseline identity");
  const auto inherited = vector::markerFor(true), enabled = scope::markerFor(true);
  require(enabled == inherited + std::string(kScopeSchema), "Enabled scope lost or changed inherited implementation marker");
  require(inherited.ends_with(kSourceID), "Inherited marker does not bind the qualified source identity");
  for (std::string_view part : {"private-R1-nonverification-decode-only-numerical-alternative",
       "direct-BF16xI8-vector4-L32O4-fourpartials-descendingXOR", "kernel=gemv_decode_sep21_v4_l32_o4",
       "cert=RNorRTZ-u23-FTZ4lambda-lateF32scale-BF16SwiGLU-v1b"})
    require(enabled.find(part) != std::string::npos, "Scope identity lost inherited producer/certificate semantics");
  require(enabled.ends_with(kScopeSchema) && enabled.find(kSourceID) != std::string::npos,
      "Scope marker omitted source or strict operation schema");
  const vector::Counters empty;
  require(!empty.enabled && !empty.gateCalls && !empty.gateRows && !empty.downCalls && !empty.downRows,
      "Graph counter snapshots do not initialize empty and disabled");
}

uint64_t selectorChecks() {
  uint64_t cases = 0, selectedCases = 0;
  for (uint32_t rows = 0; rows <= 8193; ++rows)
    for (bool standard : {false, true})
      for (bool verification : {false, true})
        for (bool gathered : {false, true})
          for (bool enabled : {false, true}) {
            ++cases;
            const bool expected = rows == 1 && standard && gathered && enabled && !verification;
            const bool actual = scope::selectedFor(standard, rows, verification, gathered, enabled);
            selectedCases += actual ? 1 : 0;
            require(actual == expected, "Selector widened standard/R1/nonverification/gathered/flag scope");
          }
  require(cases == 131104 && selectedCases == 1, "Exhaustive selector rectangle has unexpected eligible cases");
  for (bool standard : {false, true})
    for (bool verification : {false, true})
      for (bool gathered : {false, true})
        for (bool enabled : {false, true})
          require(!scope::selectedFor(standard, UINT32_MAX, verification, gathered, enabled),
              "Extreme physical row count became selected");
  static_assert(scope::selectedFor(true, 1, false, true, true));
  static_assert(!scope::selectedFor(false, 1, false, true, true));
  static_assert(!scope::selectedFor(true, 1, true, true, true));
  static_assert(!scope::selectedFor(true, 1, false, false, true));
  static_assert(!scope::selectedFor(true, 1, false, true, false));
  static_assert(!scope::selectedFor(true, 2, false, true, true));
  return cases;
}

void counterIdentityChecks(bool enabled) {
  const std::string immutable = scope::implementationMarker();
  vector::Counters snapshot;
  // Status snapshots can advance or overflow their reported values without
  // entering the source/policy identity. Actual Store atomics are source-owned
  // and independently checked by the whole-worker integration witness.
  for (uint64_t value : {uint64_t{0}, uint64_t{1}, std::numeric_limits<uint64_t>::max()}) {
    snapshot.enabled = !snapshot.enabled;
    snapshot.gateCalls = value; snapshot.gateRows = value;
    snapshot.downCalls = value; snapshot.downRows = value;
    require(snapshot.gateCalls == value && snapshot.gateRows == value && snapshot.downCalls == value && snapshot.downRows == value,
        "CPU graph-counter fixture failed to advance");
    require(scope::implementationMarker() == immutable && scope::markerFor(enabled) == immutable,
        "Mutable counter snapshots changed immutable implementation identity");
  }
}

void runtimeChecks(bool enabled) {
  require(vector::requested() == enabled, "Frozen optional policy enablement differs");
  require(scope::implementationMarker() == scope::markerFor(enabled), "Runtime implementation identity differs");
  for (uint32_t rows = 0; rows <= 8193; ++rows)
    for (bool standard : {false, true})
      for (bool verification : {false, true})
        for (bool gathered : {false, true})
          require(scope::selected(standard, rows, verification, gathered) ==
              (enabled && standard && rows == 1 && !verification && gathered),
              "Runtime selector differs from immutable explicit scope");
  require(!scope::selected(true, UINT32_MAX, false, true), "Runtime selected extreme rows");
  counterIdentityChecks(enabled);
}

void freezeChecks(std::string_view mode) {
  const char *initial = nullptr;
  bool enabled = false;
  if (mode == "--freeze0") initial = "0";
  else if (mode == "--freeze1") { initial = "1"; enabled = true; }
  else if (mode == "--missing") initial = nullptr;
  else if (mode == "--retry0" || mode == "--retry1") {
    setFlag("bad");
    requiresInvalid([] { (void)vector::requested(); }, "Malformed first initialization did not throw");
    requiresInvalid([] { (void)scope::selected(false, 0, true, false); }, "Ineligible scope ignored malformed first policy");
    requiresInvalid([] { (void)scope::implementationMarker(); }, "Identity ignored malformed first policy");
    enabled = mode == "--retry1"; initial = enabled ? "1" : "0";
  } else throw std::invalid_argument("Unknown CPU optional-state process mode");
  setFlag(initial); runtimeChecks(enabled);
  const auto frozen = vector::parseState(initial);
  for (const char *mutation : std::array<const char *, 7>{nullptr, "0", "1", "bad", " ", "1 ", "01"}) {
    if ((!initial && !mutation) || (initial && mutation && std::string_view(initial) == mutation)) continue;
    setFlag(mutation);
    const bool malformed = mutation && std::string_view(mutation) != "0" && std::string_view(mutation) != "1";
    const auto testRefusal = [&](auto operation, const char *message) {
      if (malformed) requiresInvalid(operation, message);
      else requiresChangedState(operation, message);
    };
    testRefusal([] { (void)vector::requested(); }, "Post-freeze optional-state mutation was accepted");
    testRefusal([] { (void)scope::selected(true, 1, false, true); }, "Eligible scope ignored changed frozen policy");
    testRefusal([] { (void)scope::selected(false, 0, true, false); }, "Ineligible scope bypassed strict policy drift check");
    testRefusal([] { (void)vector::eligible(2, true); }, "Inherited fallback bypassed strict policy drift check");
    testRefusal([] { (void)scope::implementationMarker(); }, "Runtime identity ignored changed frozen policy");
    setFlag(initial);
    require(vector::requested() == enabled && vector::parseState(std::getenv(vector::kFlag)) == frozen,
        "Restoring the exact frozen optional state failed");
    require(scope::implementationMarker() == scope::markerFor(enabled), "Restored optional state changed identity");
  }
  std::cout << "{\"kind\":\"gemv_r1_teacher_scope_policy_cpu\",\"mode\":\"" << mode
      << "\",\"frozen_optional_state\":\"" << stateName(frozen) << "\",\"enabled\":" << (enabled ? "true" : "false")
      << ",\"checks\":" << checks << ",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    parserChecks(); identityChecks();
    const uint64_t cases = selectorChecks();
    if (argc == 2) { freezeChecks(argv[1]); return 0; }
    if (argc != 1) throw std::invalid_argument("Use no arguments or --freeze0, --freeze1, --missing, --retry0, --retry1");
    const auto state = vector::parseState(std::getenv(vector::kFlag));
    const bool enabled = state == vector::SwitchState::Enabled;
    runtimeChecks(enabled);
    std::cout << "{\"kind\":\"gemv_r1_teacher_scope_policy_cpu\",\"selector_cases\":" << cases
        << ",\"rows_checked\":\"0..8193\",\"frozen_optional_state\":\"" << stateName(state)
        << "\",\"enabled\":" << (enabled ? "true" : "false") << ",\"source_identity_sha256\":\"" << kSourceID
        << "\",\"counter_snapshots_do_not_change_marker\":true,\"checks\":" << checks
        << ",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Teacher R1 scope CPU policy check failed: " << error.what() << '\n';
    return 1;
  }
}
