// Reuse the same guarded synthetic input construction and preregistered error
// reporting. This standalone executable only audits canonical GPU arithmetic.
#define main gdn_chunk_original_oracle_main
#include "metal_oracle.mm"
#undef main

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc == 1) {
                std::cout << "Usage: canonical-delta-oracle --quality|--resources [--library PATH] [--lanes 1..32]\n";
                return 0;
            }
            std::string mode, path = (std::filesystem::absolute(argv[0]).parent_path() /
                                      "gdn-canonical-audit.metallib").string();
            unsigned lanes = 1;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--resources" || arg == "--quality") {
                    require(mode.empty(), "choose one mode"); mode = arg;
                } else if (arg == "--library") {
                    require(++i < argc, "--library needs path"); path = argv[i];
                } else if (arg == "--lanes") {
                    require(++i < argc, "--lanes needs integer");
                    size_t used = 0; const unsigned long n = std::stoul(argv[i], &used);
                    require(used == std::strlen(argv[i]) && n && n <= 32, "invalid lanes");
                    lanes = unsigned(n);
                } else throw std::runtime_error("unknown argument: " + arg);
            }
            require(!mode.empty(), "missing mode");
            GPU gpu(path);
            auto pipeline = gpu.pipeline("private_gdn_canonical_audit_v16_t16");
            if (mode == "--resources") {
                emit(@{@"kind": @"resources", @"name": @"private_gdn_canonical_audit_v16_t16",
                       @"threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
                       @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
                       @"execution_width": @(pipeline.threadExecutionWidth)});
                return 0;
            }
            size_t ordinaryFailed = 0, stages = 0;
            const auto fixtures = gdn_chunk_cpu::make_fixtures();
            for (const auto &fixture : fixtures) {
                const bool ordinary = !rangeTrap(fixture.name);
                const auto reference = serial<double>(fixture);
                std::vector<float> seed = fixture.initial_state;
                for (unsigned continuation = 0; continuation < 2; ++continuation) {
                    const auto sequence = continuation ? gdn_chunk_cpu::continuation_fixture(19) : fixture;
                    const auto truth = continuation ? serial<double>(sequence, &reference.final_state) : reference;
                    Inputs input(gpu, sequence, lanes);
                    Results result(gpu, sequence.rows, lanes, true);
                    result.reset(seed);
                    // Canonical geometry remains 512 threads; only the loaded
                    // pipeline exposes additional stores of native registers.
                    dispatch(gpu, pipeline, Canonical, input, result, true, true);
                    input.check(); result.check(true, seed);
                    const auto history = errors(lanes * sequence.rows * V * K, truth.history, lanes,
                        [&](size_t i) { return double(result.history.as<float>()[i]); });
                    const auto delta = errors(lanes * sequence.rows * V, truth.delta, lanes,
                        [&](size_t i) { return double(result.delta.as<float>()[i]); });
                    const auto pre = errors(lanes * sequence.rows * V, truth.out, lanes,
                        [&](size_t i) { return double(result.preOutput.as<float>()[i]); });
                    const auto carried = errors(lanes * V * K, truth.final_state, lanes,
                        [&](size_t i) { return double(result.state.as<float>()[(i / (V*K))*State + i % (V*K)]); });
                    const auto output = result.headOutputs();
                    size_t castMismatch = 0;
                    for (size_t i = 0; i < output.size(); ++i)
                        castMismatch += output[i] != to_bf16(result.preOutput.as<float>()[i]);
                    const uint32_t diagnostics = *result.diagnostics.as<uint32_t>();
                    const bool pass = history.pass() && delta.pass() && pre.pass() && carried.pass() &&
                                      !castMismatch && !diagnostics;
                    ordinaryFailed += ordinary && !pass; ++stages;
                    emit(@{@"kind": @"canonical_delta_quality", @"fixture": ns(fixture.name),
                           @"stage": continuation ? @"continuation" : @"initial", @"rows": @(sequence.rows),
                           @"lanes": @(lanes), @"classification": ordinary ? @"ordinary" : @"range_trap",
                           @"required": @(ordinary), @"preregistered_all_fields_pass": @(pass),
                           @"history_f32": history.json(), @"delta_f32": delta.json(),
                           @"pre_bf16_output_f32": pre.json(), @"carried_state_f32": carried.json(),
                           @"bf16_mismatch_own_f32_count": @(castMismatch), @"diagnostics": @(diagnostics),
                           @"canaries_pass": @YES, @"immutable_sha256_pass": @YES});
                    seed = result.carried();
                }
            }
            emit(@{@"kind": @"canonical_delta_summary", @"stages": @(stages),
                   @"ordinary_failed_stages": @(ordinaryFailed),
                   @"preregistered_all_fields_pass": @(ordinaryFailed == 0),
                   @"scope": @"canonical native SIMD arithmetic, separate audit stores only"});
            return ordinaryFailed ? 2 : 0;
        } catch (const std::exception &e) {
            emit(@{@"kind": @"error", @"message": ns(e.what())});
            return 1;
        }
    }
}
