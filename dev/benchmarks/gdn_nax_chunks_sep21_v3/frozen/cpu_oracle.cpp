#include "cpu_oracle.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>

namespace gdn_chunk_cpu {
namespace {

struct RNG {
    uint64_t state;
    explicit RNG(uint64_t seed) : state(seed ? seed : 1) {}
    uint32_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return uint32_t((state * 2685821657736338717ULL) >> 32);
    }
    float unit() { return float(next() >> 8) * (1.0f / 16777216.0f); }
    float signed_unit() { return 2.0f * unit() - 1.0f; }
};

Fixture random_fixture(const std::string& name, uint64_t seed, size_t rows = 67) {
    Fixture f;
    f.name = name;
    f.rows = rows;
    f.q.resize(rows * f.key_dim);
    f.k.resize(rows * f.key_dim);
    f.v.resize(rows * f.value_dim);
    f.beta.resize(rows);
    f.alpha.resize(rows);
    f.initial_state.resize(f.value_dim * f.key_dim);
    RNG rng(seed);
    for (float& x : f.initial_state) x = rng.signed_unit() * 0.08f;
    std::vector<float> row(f.key_dim);
    for (size_t t = 0; t < rows; ++t) {
        for (auto* tensor : {&f.q, &f.k}) {
            double norm2 = 0;
            for (float& x : row) {
                x = rng.signed_unit();
                norm2 += double(x) * double(x);
            }
            const float invnorm = float(1.0 / std::sqrt(norm2));
            for (size_t k = 0; k < f.key_dim; ++k)
                (*tensor)[t * f.key_dim + k] = to_bf16(row[k] * invnorm);
        }
        for (size_t v = 0; v < f.value_dim; ++v)
            f.v[t * f.value_dim + v] = to_bf16(rng.signed_unit());
        f.beta[t] = to_bf16(0.05f + 0.95f * rng.unit());
        f.alpha[t] = 0.85f + 0.15f * rng.unit();
    }
    return f;
}

Fixture one_hot_fixture() {
    Fixture f = random_fixture("one_hot_orientation", 101);
    const size_t K = f.key_dim, V = f.value_dim;
    std::fill(f.q.begin(), f.q.end(), to_bf16(0));
    std::fill(f.k.begin(), f.k.end(), to_bf16(0));
    for (size_t v = 0; v < V; ++v)
        for (size_t k = 0; k < K; ++k)
            f.initial_state[v * K + k] = float(int((v * 19 + k * 7) % 33) - 16) / 32;
    for (size_t t = 0; t < f.rows; ++t) {
        const size_t ki = (t * 3 + 2) % 11, qi = (t * 7 + 1) % 11;
        f.k[t * K + ki] = to_bf16(t % 3 ? 1.0f : -1.0f);
        f.q[t * K + qi] = to_bf16(t % 2 ? 0.5f : -1.0f);
        f.alpha[t] = t % 13 == 4 ? 0.0f : (t % 3 ? 1.0f : 0.5f);
        f.beta[t] = to_bf16(float(t % 5) / 4);
        for (size_t v = 0; v < V; ++v)
            f.v[t * V + v] = to_bf16(float(int((t * 5 + v * 3) % 17) - 8) / 16);
    }
    return f;
}

Fixture cancellation_fixture() {
    Fixture f = random_fixture("cancellation_repeated_key", 404);
    const size_t K = f.key_dim, V = f.value_dim;
    // Exactly unit norm: 64 coordinates of magnitude 1/8.
    for (size_t t = 0; t < f.rows; ++t) {
        f.alpha[t] = 1;
        f.beta[t] = to_bf16(1);
        for (size_t k = 0; k < K; ++k) {
            f.k[t * K + k] = to_bf16(k < 64 ? (k % 2 ? -0.125f : 0.125f) : 0.0f);
            f.q[t * K + k] = f.k[t * K + k];
        }
    }
    std::vector<double> state(f.initial_state.begin(), f.initial_state.end());
    for (size_t t = 0; t < f.rows; ++t)
        for (size_t v = 0; v < V; ++v) {
            double memory = 0;
            for (size_t k = 0; k < K; ++k)
                memory += state[v * K + k] * double(from_bf16(f.k[t * K + k]));
            // BF16 rounding leaves small residuals after near-cancellation.
            const float perturbation = float(int((t + v) % 3) - 1) * 0.000001f;
            f.v[t * V + v] = to_bf16(float(memory) + perturbation);
            const double delta = double(from_bf16(f.v[t * V + v])) - memory;
            for (size_t k = 0; k < K; ++k)
                state[v * K + k] += delta * double(from_bf16(f.k[t * K + k]));
        }
    return f;
}

Fixture prefix_update_fixture() {
    Fixture f = one_hot_fixture();
    f.name = "prefix_underflow_surviving_update";
    std::fill(f.initial_state.begin(), f.initial_state.end(), 0);
    std::fill(f.q.begin(), f.q.end(), to_bf16(0));
    std::fill(f.k.begin(), f.k.end(), to_bf16(0));
    std::fill(f.v.begin(), f.v.end(), to_bf16(0));
    std::fill(f.beta.begin(), f.beta.end(), to_bf16(0));
    std::fill(f.alpha.begin(), f.alpha.end(), 1.0f);
    for (size_t t = 0; t < f.rows; ++t) {
        f.q[t * f.key_dim] = to_bf16(1);
        f.k[t * f.key_dim] = to_bf16(1);
    }
    f.alpha[0] = std::ldexp(1.0f, -100);
    f.alpha[1] = std::ldexp(1.0f, -100);
    f.alpha[2] = 0.5f;
    f.beta[1] = to_bf16(1);
    for (size_t v = 0; v < f.value_dim; ++v)
        f.v[f.value_dim + v] = to_bf16(float(int(v % 9) - 4) / 4);
    return f;
}

Fixture range_fixture() {
    Fixture f = prefix_update_fixture();
    f.name = "extreme_range_prefix_underflow";
    std::fill(f.beta.begin(), f.beta.end(), to_bf16(0));
    f.alpha[2] = 1;
    for (size_t v = 0; v < f.value_dim; ++v)
        f.initial_state[v * f.key_dim] = std::ldexp(1.0f, 100);
    return f;
}

} // namespace

Fixture prefix_fixture(const Fixture& f, size_t rows) {
    if (!rows || rows > f.rows) throw std::runtime_error("Invalid prefix length");
    Fixture result = f;
    result.name += "_rows" + std::to_string(rows);
    result.rows = rows;
    result.q.resize(rows * f.key_dim);
    result.k.resize(rows * f.key_dim);
    result.v.resize(rows * f.value_dim);
    result.beta.resize(rows);
    result.alpha.resize(rows);
    return result;
}

Fixture continuation_fixture(size_t rows) {
    return random_fixture("continued_sequence", 0xabc1007, rows);
}

std::vector<Fixture> make_fixtures() {
    std::vector<Fixture> fs;
    for (uint64_t seed : {uint64_t(0x91821), uint64_t(0x91822), uint64_t(0x91823)})
        fs.push_back(random_fixture("random_normalized_" + std::to_string(seed), seed));
    Fixture base = fs.front();
    Fixture zeros = base;
    zeros.name = "decay_zero_boundaries";
    for (size_t t : {size_t(0), size_t(14), size_t(15), size_t(16), size_t(31),
                     size_t(32), size_t(63), size_t(66)}) zeros.alpha[t] = 0;
    fs.push_back(std::move(zeros));
    Fixture tiny = base;
    tiny.name = "decay_f32_prefix_underflow";
    std::fill(tiny.alpha.begin(), tiny.alpha.end(), 0.0001f);
    fs.push_back(std::move(tiny));
    Fixture extreme_decay = base;
    extreme_decay.name = "decay_f64_prefix_underflow";
    std::fill(extreme_decay.alpha.begin(), extreme_decay.alpha.end(), 1.0e-20f);
    fs.push_back(std::move(extreme_decay));
    Fixture beta_zero = base;
    beta_zero.name = "beta_zero";
    std::fill(beta_zero.beta.begin(), beta_zero.beta.end(), to_bf16(0));
    fs.push_back(std::move(beta_zero));
    Fixture beta_one = base;
    beta_one.name = "beta_one";
    std::fill(beta_one.beta.begin(), beta_one.beta.end(), to_bf16(1));
    fs.push_back(std::move(beta_one));
    Fixture alpha_one = base;
    alpha_one.name = "alpha_one";
    std::fill(alpha_one.alpha.begin(), alpha_one.alpha.end(), 1);
    fs.push_back(std::move(alpha_one));
    fs.push_back(cancellation_fixture());
    fs.push_back(one_hot_fixture());
    fs.push_back(prefix_update_fixture());
    fs.push_back(range_fixture());
    for (size_t rows : {size_t(1), size_t(15), size_t(16), size_t(17), size_t(31),
                        size_t(32), size_t(33), size_t(63), size_t(64), size_t(65)})
        fs.push_back(prefix_fixture(base, rows));
    return fs;
}

} // namespace gdn_chunk_cpu

#ifndef GDN_CPU_ORACLE_NO_MAIN
namespace {
using namespace gdn_chunk_cpu;

struct Metric {
    size_t count = 0, differing = 0, nonzero_lost = 0;
    bool finite = true;
    double max_abs = 0, sum_error2 = 0, sum_reference2 = 0, max_scaled = 0;
    size_t max_abs_index = 0;
    void merge(const Metric& other) {
        if (other.max_abs > max_abs) max_abs_index = count + other.max_abs_index;
        count += other.count;
        differing += other.differing;
        nonzero_lost += other.nonzero_lost;
        finite &= other.finite;
        max_abs = std::max(max_abs, other.max_abs);
        max_scaled = std::max(max_scaled, other.max_scaled);
        sum_error2 += other.sum_error2;
        sum_reference2 += other.sum_reference2;
    }
    bool pass() const { return finite && max_scaled <= 1; }
};

template <typename A, typename B>
Metric compare(const std::vector<A>& ref, const std::vector<B>& got, double atol, double rtol) {
    if (ref.size() != got.size()) throw std::runtime_error("Comparison length mismatch");
    Metric m;
    m.count = ref.size();
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = double(ref[i]), g = double(got[i]);
        if (!std::isfinite(r) || !std::isfinite(g)) {
            m.finite = false;
            continue;
        }
        const double e = std::abs(r - g);
        m.differing += r != g;
        m.nonzero_lost += r != 0 && g == 0;
        if (e > m.max_abs) { m.max_abs = e; m.max_abs_index = i; }
        m.sum_error2 += e * e;
        m.sum_reference2 += r * r;
        const double denom = atol + rtol * std::max(std::abs(r), std::abs(g));
        const double scaled = denom ? e / denom : (e ? std::numeric_limits<double>::infinity() : 0);
        m.max_scaled = std::max(m.max_scaled, scaled);
    }
    return m;
}

void number(double x) {
    if (std::isfinite(x)) std::cout << x;
    else std::cout << "null";
}

void emit_metric(const Metric& m) {
    std::cout << "{\"count\":" << m.count << ",\"finite\":" << (m.finite ? "true" : "false")
              << ",\"pass\":" << (m.pass() ? "true" : "false") << ",\"max_abs\":";
    number(m.max_abs);
    std::cout << ",\"max_abs_index\":" << m.max_abs_index << ",\"rms_abs\":";
    number(m.count ? std::sqrt(m.sum_error2 / double(m.count)) : 0);
    std::cout << ",\"relative_l2\":";
    number(m.sum_reference2 ? std::sqrt(m.sum_error2 / m.sum_reference2) :
           (m.sum_error2 ? std::numeric_limits<double>::infinity() : 0));
    std::cout << ",\"max_scaled_error\":";
    number(m.max_scaled);
    std::cout << ",\"different_elements\":" << m.differing
              << ",\"reference_nonzero_candidate_zero\":" << m.nonzero_lost << "}";
}

template <typename A, typename B>
Metric emit_comparison(const Trace<A>& ref, const Trace<B>& got, double atol, double rtol) {
    std::cout << "{";
    Metric total;
    const char* names[] = {"memory", "delta", "output", "history", "final_state"};
    const std::vector<A>* refs[] = {&ref.memory, &ref.delta, &ref.out, &ref.history, &ref.final_state};
    const std::vector<B>* gots[] = {&got.memory, &got.delta, &got.out, &got.history, &got.final_state};
    for (size_t i = 0; i < 5; ++i) {
        if (i) std::cout << ',';
        std::cout << '"' << names[i] << "\":";
        const Metric m = compare(*refs[i], *gots[i], atol, rtol);
        emit_metric(m);
        total.merge(m);
    }
    std::cout << ",\"all_fields\":";
    emit_metric(total);
    std::cout << "}";
    return total;
}

template <typename T> bool exact_retained_update(const Trace<T>& r, const Fixture& f) {
    for (size_t t = 2; t < f.rows; ++t)
        for (size_t v = 0; v < f.value_dim; ++v)
            if (r.out[t * f.value_dim + v] != T(from_bf16(f.v[f.value_dim + v])) * T(0.5))
                return false;
    return true;
}

template <typename T> bool exact_retained_range(const Trace<T>& r, const Fixture& f) {
    for (size_t t = 1; t < f.rows; ++t)
        for (size_t v = 0; v < f.value_dim; ++v)
            if (r.out[t * f.value_dim + v] != std::ldexp(T(1), -100)) return false;
    return true;
}

} // namespace

int main() {
    constexpr double d_atol = 1e-11, d_rtol = 1e-10;
    constexpr double f_atol = 2e-5, f_rtol = 2e-4;
    const auto fixtures = make_fixtures();
    const auto continued = continuation_fixture();
    Metric aggregate64, aggregate32, aggregate32_vs64;
    size_t cases = 0, f64_failures = 0, ordinary_f32_failures = 0, strict_failures = 0;
    size_t expected_f32_range_losses = 0;
    std::cout << std::setprecision(17);
    std::cout << "{\"schema\":\"gdn_chunk_cpu_oracle_v1\",\"source_types\":{\"q_k_v_beta\":\"BF16_RNE\","
                 "\"alpha_initial_state\":\"F32\"},\"dimensions\":{\"K\":128,\"V\":128},"
                 "\"arithmetic\":{\"serial\":\"elementwise state decay; scalar memory dot; delta; rank-one update; state-query dot\","
                 "\"chunk\":\"direct relative products; distributed-beta triangular solve; algebraic state/output reconstruction\","
                 "\"fma\":\"build with -ffp-contract=off; no fast-math\"},"
                 "\"tolerances\":{\"F64\":{\"absolute\":" << d_atol << ",\"relative\":" << d_rtol << "},"
                 "\"F32\":{\"absolute\":" << f_atol << ",\"relative\":" << f_rtol << "}},\"cases\":[";
    for (const Fixture& f : fixtures) {
        const auto ref64 = serial<double>(f);
        const auto ref32 = serial<float>(f);
        const auto continued64 = serial<double>(continued, &ref64.final_state);
        const auto continued32 = serial<float>(continued, &ref32.final_state);
        for (size_t C : {size_t(16), size_t(32)}) {
            const auto candidate64 = chunked<double>(f, C);
            const auto candidate32 = chunked<float>(f, C);
            const auto after64 = serial<double>(continued, &candidate64.final_state);
            const auto after32 = serial<float>(continued, &candidate32.final_state);
            if (cases++) std::cout << ',';
            std::cout << "{\"fixture\":\"" << f.name << "\",\"rows\":" << f.rows
                      << ",\"chunk_size\":" << C << ",\"continued_rows\":" << continued.rows
                      << ",\"f64_equivalence\":";
            Metric m64 = emit_comparison(ref64, candidate64, d_atol, d_rtol);
            std::cout << ",\"f64_continued_equivalence\":";
            m64.merge(emit_comparison(continued64, after64, d_atol, d_rtol));
            aggregate64.merge(m64);
            f64_failures += !m64.pass();
            std::cout << ",\"f32_association_vs_serial_f32\":";
            Metric m32 = emit_comparison(ref32, candidate32, f_atol, f_rtol);
            std::cout << ",\"f32_continued_association\":";
            m32.merge(emit_comparison(continued32, after32, f_atol, f_rtol));
            aggregate32.merge(m32);
            ordinary_f32_failures += !m32.pass() && f.name != "extreme_range_prefix_underflow";
            std::cout << ",\"serial_f32_vs_serial_f64\":";
            emit_comparison(ref64, ref32, f_atol, f_rtol);
            std::cout << ",\"chunk_f32_vs_serial_f64\":";
            aggregate32_vs64.merge(emit_comparison(ref64, candidate32, f_atol, f_rtol));
            std::cout << ",\"strict_checks\":{";
            if (f.name == "prefix_underflow_surviving_update") {
                const bool ok64 = exact_retained_update(candidate64, f), ok32 = exact_retained_update(candidate32, f);
                strict_failures += !ok64 || !ok32;
                std::cout << "\"fresh_update_f64_exact\":" << (ok64 ? "true" : "false")
                          << ",\"fresh_update_f32_exact\":" << (ok32 ? "true" : "false");
            } else if (f.name == "extreme_range_prefix_underflow") {
                const bool serial_ok = exact_retained_range(ref32, f);
                const bool ok64 = exact_retained_range(candidate64, f), ok32 = exact_retained_range(candidate32, f);
                strict_failures += !serial_ok || !ok64;
                expected_f32_range_losses += !ok32;
                std::cout << "\"serial_f32_range_retained\":" << (serial_ok ? "true" : "false")
                          << ",\"chunk_f64_range_retained\":" << (ok64 ? "true" : "false")
                          << ",\"chunk_f32_range_retained\":" << (ok32 ? "true" : "false")
                          << ",\"classification\":\"expected_F32_prefix_underflow_association_loss\"";
            } else {
                std::cout << "\"classification\":\"bounded_roundoff_fixture\"";
            }
            std::cout << "}}";
        }
    }
    std::cout << "],\"summary\":{\"case_count\":" << cases << ",\"fixture_count\":" << fixtures.size()
              << ",\"f64_equivalence_failures\":" << f64_failures
              << ",\"ordinary_f32_tolerance_failures\":" << ordinary_f32_failures
              << ",\"strict_invariant_failures\":" << strict_failures
              << ",\"expected_f32_range_loss_cases\":" << expected_f32_range_losses
              << ",\"f64_all_fields_and_continuation\":";
    emit_metric(aggregate64);
    std::cout << ",\"f32_association_all_fields_and_continuation\":";
    emit_metric(aggregate32);
    std::cout << ",\"chunk_f32_vs_serial_f64_all_fields\":";
    emit_metric(aggregate32_vs64);
    const bool pass = !f64_failures && !ordinary_f32_failures && !strict_failures;
    std::cout << ",\"pass\":" << (pass ? "true" : "false") << "}}\n";
    return pass ? 0 : 1;
}
#endif
