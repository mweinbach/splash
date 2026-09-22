#include "worker_bridge.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>

namespace policy = splash::flash::gdn_prefill_fma_sep21;

namespace {
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template <typename F> void requiresInvalidArgument(F&& f, const char* message) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, message);
}

void set(const char* key, const char* value) {
    const int result = value ? setenv(key, value, 1) : unsetenv(key);
    if (result != 0) throw std::runtime_error("Could not set test process environment");
}

bool isDigest(std::string_view value) {
    if (value.size() != 64) return false;
    for (char c : value) if (!(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f')) return false;
    return true;
}

void parserTests() {
    const std::array<const char*, 16> invalidFlags{
        "", "00", "01", "+1", "-0", "-1", "2", "true", "false", "yes", " 1", "1 ",
        "0\n", "1\n", "0\t", "1\t"};
    for (const char* raw : invalidFlags) {
        requiresInvalidArgument([&] { (void)policy::detail::parseFlag(raw); }, "Malformed opt-in accepted");
        requiresInvalidArgument([&] { (void)policy::detail::parseRequested(raw, "1"); }, "Malformed request accepted");
    }
    const std::array<const char*, 12> stagedValues{
        nullptr, "", "0", "1", "bad", "true", "false", "01", " 1", "1 ", "1\n", "+1"};
    for (const char* staged : stagedValues) {
        require(!policy::detail::parseRequested(nullptr, staged), "Missing opt-in parsed staged dependency");
        require(!policy::detail::parseRequested("0", staged), "Disabled opt-in parsed staged dependency");
        if (staged && std::strcmp(staged, "1") == 0)
            require(policy::detail::parseRequested("1", staged), "Valid staged dependency rejected");
        else
            requiresInvalidArgument([&] { (void)policy::detail::parseRequested("1", staged); }, "Invalid staged dependency accepted");
    }
    set(policy::kFlag, nullptr); set(policy::kStagedFlag, "bad");
    require(!policy::detail::requestedFromEnvironment(), "Missing environment request was not false");
    set(policy::kFlag, "0");
    require(!policy::detail::requestedFromEnvironment(), "Disabled environment request parsed bad staged value");
    set(policy::kFlag, "1");
    requiresInvalidArgument([] { (void)policy::detail::requestedFromEnvironment(); }, "Enabled request ignored bad staged value");
    set(policy::kStagedFlag, "1");
    require(policy::detail::requestedFromEnvironment(), "Enabled environment request rejected");
    set(policy::kFlag, "");
    requiresInvalidArgument([] { (void)policy::detail::requestedFromEnvironment(); }, "Empty environment request accepted");
    set(policy::kFlag, nullptr); set(policy::kStagedFlag, nullptr);
}

size_t routeTests() {
    struct Native { const char* pipeline; uint32_t values, time; };
    constexpr std::array<Native, 4> natives{{
        {"flash_gdn_staged_v8_t16", 8, 16}, {"flash_gdn_staged_v8_t32", 8, 32},
        {"flash_gdn_staged_v16_t16", 16, 16}, {"flash_gdn_staged_v16_t32", 16, 32}}};
    size_t cases = 0;
    for (size_t rows = 0; rows <= 2049; ++rows)
        for (size_t lanes = 0; lanes <= 33; ++lanes)
            for (bool enabled : std::array<bool, 2>{false, true}) {
                const bool expected = enabled && rows >= 64 && rows <= 2048 && lanes >= 1 && lanes <= 32;
                require(policy::eligible(rows, lanes, enabled) == expected, "Eligibility differs from qualified rectangle");
                for (const auto& native : natives) {
                    const auto selected = policy::route(native.pipeline, native.values, native.time, rows, lanes, enabled);
                    ++cases;
                    require(selected.usesPrivatePrefill == expected, "Route enablement differs from eligibility");
                    if (expected) {
                        require(selected.pipeline == policy::kPipeline && selected.values == 16 && selected.time == 32,
                                "Enabled route chose the wrong pipeline or tile");
                    } else {
                        require(selected.pipeline == native.pipeline && selected.values == native.values && selected.time == native.time,
                                "Native route changed while ineligible or disabled");
                    }
                }
            }
    const size_t extreme = std::numeric_limits<size_t>::max();
    for (size_t rows : std::array<size_t, 3>{0, 2049, extreme})
        for (size_t lanes : std::array<size_t, 3>{0, 33, extreme})
            require(!policy::eligible(rows, lanes, true), "Extreme integer became eligible");
    require(!policy::eligible(extreme, 1, true), "Maximum row count became eligible");
    require(!policy::eligible(64, extreme, true), "Maximum lane count became eligible");
    static_assert(policy::eligible(64, 1, true));
    static_assert(policy::eligible(2048, 32, true));
    static_assert(!policy::eligible(63, 1, true));
    static_assert(!policy::eligible(2049, 32, true));
    static_assert(!policy::eligible(64, 0, true));
    static_assert(!policy::eligible(2048, 33, true));
    static_assert(!policy::eligible(64, 1, false));
    static_assert(policy::kGPUAddedAllocations == 0);
    return cases;
}

void identityTests(const std::string& sourcePath) {
    require(policy::detail::sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA256 empty-vector mismatch");
    require(policy::detail::sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA256 abc-vector mismatch");
    const std::string base = "native-fixture-policy-v1";
    // Independently computed with Python hashlib over the specified bytes.
    require(policy::numericalIdentity(base, true) == "6cfda381257b3dced629a9b62a630545b9cc46adb0b989d383f78adba4883c9a",
            "Source-bound numerical identity golden mismatch");
    const std::array<std::string, 5> bases{std::string(), base, "native\nidentity\n", std::string("native\0identity", 15), "0"};
    for (const std::string& value : bases) {
        require(policy::numericalIdentity(value, false) == value, "Disabled numerical identity changed native bytes");
        require(policy::operationNumericalIdentity(value, false) == value, "Disabled operation identity changed native bytes");
        const auto enabled = policy::numericalIdentity(value, true);
        require(isDigest(enabled), "Enabled numerical identity is not lowercase SHA256");
        require(enabled == policy::numericalIdentity(value, true), "Enabled numerical identity is nondeterministic");
        require(enabled == policy::operationNumericalIdentity(value, true), "Operation identity alias differs");
        std::string expectedBytes = value + '\n' + policy::kPolicy + '\n' + policy::kKernelSourceSHA256;
        require(enabled == policy::detail::sha256(expectedBytes), "Identity bytes do not match the specified construction");
    }
    require(policy::numericalIdentity(base, true) != policy::numericalIdentity(base + "x", true), "Identity does not bind base");
    require(std::strcmp(policy::marker(false), "") == 0, "Disabled marker is not empty");
    require(std::strcmp(policy::marker(true), policy::kMarker) == 0, "Enabled marker differs from policy marker");
    require(std::string_view(policy::marker(true)).find(";private-gdn-prefill-fma-v16-t32-rows64to2048") == 0,
            "Enabled marker is not distinctive");
    const char* nativeMarker = ";private-gdn-staged-native-fixture";
    require(policy::nativeStageMarker(nativeMarker, false) == nativeMarker, "Disabled stage marker changed native pointer");
    require(policy::nativeStageMarker(nativeMarker, true) == policy::kMarker, "Enabled stage marker was not replaced");
    require(isDigest(policy::kKernelSourceSHA256), "Pinned kernel source SHA256 is malformed");
    std::ifstream source(sourcePath, std::ios::binary);
    if (!source) throw std::runtime_error("Cannot open scalar_fma.metal source for immutable SHA check");
    const std::string bytes{std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
    require(policy::detail::sha256(bytes) == policy::kKernelSourceSHA256, "Pinned qualified kernel source SHA256 changed");
}

void freezeTests(std::string_view mode) {
    bool expected;
    if (mode == "--freeze0") {
        set(policy::kFlag, "0"); set(policy::kStagedFlag, "bad"); expected = false;
    } else if (mode == "--freeze1") {
        set(policy::kFlag, "1"); set(policy::kStagedFlag, "1"); expected = true;
    } else if (mode == "--missing0") {
        set(policy::kFlag, nullptr); set(policy::kStagedFlag, "bad"); expected = false;
    } else if (mode == "--retry0") {
        set(policy::kFlag, "bad"); set(policy::kStagedFlag, "1");
        requiresInvalidArgument([] { (void)policy::requested(); }, "Bad first request did not throw");
        set(policy::kFlag, "0"); set(policy::kStagedFlag, "bad"); expected = false;
    } else if (mode == "--retry1") {
        set(policy::kFlag, "1"); set(policy::kStagedFlag, "bad");
        requiresInvalidArgument([] { (void)policy::requested(); }, "Bad first dependency did not throw");
        set(policy::kStagedFlag, "1"); expected = true;
    } else throw std::invalid_argument("Unknown freeze test mode");
    require(policy::requested() == expected, "First successful frozen request differs");
    set(policy::kFlag, expected ? "0" : "1"); set(policy::kStagedFlag, expected ? "bad" : "1");
    require(policy::requested() == expected, "Environment mutation changed frozen request");
    set(policy::kFlag, ""); set(policy::kStagedFlag, nullptr);
    require(policy::requested() == expected, "Malformed later environment changed frozen request");
    set(policy::kFlag, nullptr); set(policy::kStagedFlag, "bad");
    require(policy::requested() == expected, "Missing later environment changed frozen request");
    std::cout << "{\"kind\":\"frozen_prefill_policy\",\"mode\":\"" << mode
              << "\",\"enabled\":" << (expected ? "true" : "false") << ",\"checks\":" << checks << ",\"pass\":true}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            freezeTests(argv[1]);
            return 0;
        }
        std::string sourcePath = "dev/benchmarks/gdn_chunk_sep21/scalar_fma.metal";
        if (argc == 3 && std::strcmp(argv[1], "--source") == 0) sourcePath = argv[2];
        else if (argc != 1) throw std::invalid_argument("Use no arguments, --source <scalar_fma.metal>, or a freeze test mode");
        parserTests();
        const size_t routeCases = routeTests();
        identityTests(sourcePath);
        std::cout << "{\"kind\":\"gdn_prefill_fma_worker_policy_cpu\",\"route_cases\":" << routeCases
                  << ",\"rows_checked\":\"0..2049\",\"lanes_checked\":\"0..33\",\"native_selectors\":4,"
                     "\"enabled_and_disabled\":true,\"GPU_added_allocations\":" << policy::kGPUAddedAllocations
                  << ",\"kernel_source_sha256\":\"" << policy::kKernelSourceSHA256
                  << "\",\"checks\":" << checks << ",\"pass\":true}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Worker prefill policy CPU test failed: " << e.what() << '\n';
        return 1;
    }
}
