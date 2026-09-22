#include "wy_cpu.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
using namespace gdn_chunk_cpu;
constexpr double F64AbsoluteTolerance = 1e-12;
// Unchanged gates from gdn_chunk_sep21/metal_oracle_prepared.mm.
constexpr double F32RelativeTolerance = 1e-4, F32AbsoluteScaleTolerance = 5e-4;

struct Metric {
    size_t count = 0, nonfinite = 0, differing = 0, nonzero_lost = 0;
    double max_abs = 0, reference_peak = 0;
    long double error2 = 0, reference2 = 0;
    double relative_l2() const {
        return reference2 ? double(std::sqrt(error2 / reference2)) :
            (error2 ? std::numeric_limits<double>::infinity() : 0);
    }
    bool f64_pass() const { return !nonfinite && max_abs <= F64AbsoluteTolerance; }
    bool f32_pass() const {
        return !nonfinite && relative_l2() <= F32RelativeTolerance &&
            max_abs <= F32AbsoluteScaleTolerance * std::max(1.0, reference_peak);
    }
    void merge(const Metric& m) {
        count += m.count; nonfinite += m.nonfinite; differing += m.differing;
        nonzero_lost += m.nonzero_lost;
        max_abs = std::max(max_abs, m.max_abs);
        reference_peak = std::max(reference_peak, m.reference_peak);
        error2 += m.error2; reference2 += m.reference2;
    }
};

template <typename A, typename B> Metric compare(const std::vector<A>& ref,
                                                const std::vector<B>& got) {
    if (ref.size() != got.size()) throw std::runtime_error("WY comparison dimension mismatch");
    Metric m; m.count = ref.size();
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = double(ref[i]), g = double(got[i]);
        if (!std::isfinite(r) || !std::isfinite(g)) { ++m.nonfinite; continue; }
        const double error = std::abs(r - g);
        m.max_abs = std::max(m.max_abs, error);
        m.reference_peak = std::max(m.reference_peak, std::abs(r));
        m.error2 += static_cast<long double>(error) * error;
        m.reference2 += static_cast<long double>(r) * r;
        m.differing += r != g;
        m.nonzero_lost += r != 0 && g == 0;
    }
    return m;
}

void number(double x) {
    if (std::isfinite(x)) std::cout << x; else std::cout << "null";
}
void emit_metric(const Metric& m, bool f64) {
    std::cout << "{\"count\":" << m.count << ",\"nonfinite\":" << m.nonfinite
              << ",\"max_abs\":";
    number(m.max_abs);
    std::cout << ",\"rms_abs\":";
    number(m.count ? double(std::sqrt(m.error2 / m.count)) : 0);
    std::cout << ",\"relative_l2\":";
    number(m.relative_l2());
    std::cout << ",\"reference_peak\":";
    number(m.reference_peak);
    std::cout << ",\"max_abs_tolerance\":";
    number(f64 ? F64AbsoluteTolerance : F32AbsoluteScaleTolerance * std::max(1.0, m.reference_peak));
    if (!f64) std::cout << ",\"relative_l2_tolerance\":" << F32RelativeTolerance;
    std::cout << ",\"different_elements\":" << m.differing
              << ",\"reference_nonzero_candidate_zero\":" << m.nonzero_lost
              << ",\"pass\":" << ((f64 ? m.f64_pass() : m.f32_pass()) ? "true" : "false") << '}';
}

struct Comparison {
    Metric aggregate;
    bool pass = true;
};
template <typename A, typename B> Comparison emit_comparison(const Trace<A>& ref,
                                                            const Trace<B>& got, bool f64) {
    const char* names[] = {"memory", "delta", "output", "history", "final_state"};
    const std::vector<A>* refs[] = {&ref.memory, &ref.delta, &ref.out, &ref.history, &ref.final_state};
    const std::vector<B>* gots[] = {&got.memory, &got.delta, &got.out, &got.history, &got.final_state};
    Comparison c;
    std::cout << '{';
    for (size_t i = 0; i < 5; ++i) {
        if (i) std::cout << ',';
        const auto m = compare(*refs[i], *gots[i]);
        c.aggregate.merge(m); c.pass &= f64 ? m.f64_pass() : m.f32_pass();
        std::cout << '"' << names[i] << "\":";
        emit_metric(m, f64);
    }
    std::cout << ",\"all_fields\":";
    emit_metric(c.aggregate, f64);
    std::cout << ",\"every_field_pass\":" << (c.pass ? "true" : "false") << '}';
    return c;
}

template <typename T> bool padded_zero(const gdn_nax_cpu::Prepared<T>& p) {
    for (size_t chunk = 0; chunk < p.num_chunks; ++chunk) {
        const size_t C = p.active_rows(chunk), Trows = p.chunk_size;
        for (size_t i = 0; i < Trows; ++i) {
            if (i >= C) {
                if (p.prefix(chunk)[i] != 0) return false;
                for (size_t k = 0; k < p.key_dim; ++k)
                    if (p.w(chunk)[i * p.key_dim + k] != 0 || p.e(chunk)[i * p.key_dim + k] != 0)
                        return false;
                for (size_t v = 0; v < p.value_dim; ++v)
                    if (p.u(chunk)[i * p.value_dim + v] != 0) return false;
            }
            for (size_t j = 0; j < Trows; ++j)
                if ((i >= C || j >= C || j > i) && p.score(chunk)[i * Trows + j] != 0)
                    return false;
        }
    }
    return true;
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

std::string quote(const std::string& value) {
    std::ostringstream s; s << '"';
    for (const char c : value) {
        if (c == '"' || c == '\\') s << '\\';
        s << c;
    }
    s << '"'; return s.str();
}
} // namespace

int main() {
    try {
        const auto fixtures = make_fixtures();
        const auto continuation = continuation_fixture();
        size_t cases = 0, f64_failures = 0, f32_failures = 0, non_range_f32_failures = 0;
        size_t ordinary_f32_failures = 0;
        size_t cancellation_failures = 0, strict_failures = 0, expected_f32_range_losses = 0;
        size_t padding_failures = 0;
        Metric all64, all32, all32_association;
        std::cout << std::setprecision(17)
                  << "{\"schema\":\"gdn_wy_cpu_reference_v1\",\"scope\":\"CPU_only_no_Metal_GPU_model_payload_or_production_runtime\","
                     "\"source_types\":{\"q_k_v_beta\":\"BF16_RNE\",\"alpha_initial_state\":\"F32\"},"
                     "\"dimensions\":{\"K\":128,\"V\":128},"
                     "\"prepared_layout\":\"per_chunk_row_major_W(T*128),U(T*128),E(T*128),score(T*T),prefix(T);zero_padded_tail\","
                     "\"arithmetic\":{\"prepare\":\"direct_relative_decay_products;F=(I+L)^-1;W=F*diag(beta*prefix)*K;U=F*diag(beta)*V\","
                     "\"execute\":\"D=U-W*S0^T;output=diag(prefix)*Q*S0^T+score*D;S_end=prefix_end*S0+D^T*E\","
                     "\"fma\":\"ffp_contract_off;no_fast_math\","
                     "\"qualification\":\"scalar_CPU_association_baseline_not_GPU_numerical_certificate\"},"
                     "\"tolerances\":{\"F64_max_absolute\":" << F64AbsoluteTolerance
                  << ",\"F32_relative_l2\":" << F32RelativeTolerance
                  << ",\"F32_max_absolute_scale\":" << F32AbsoluteScaleTolerance << "},\"cases\":[";
        for (const auto& f : fixtures) {
            const auto ref64 = serial<double>(f);
            const auto ref32 = serial<float>(f);
            const auto continued64 = serial<double>(continuation, &ref64.final_state);
            const auto continued32 = serial<float>(continuation, &ref32.final_state);
            for (const size_t C : {size_t(16), size_t(32)}) {
                const auto p64 = gdn_nax_cpu::prepare<double>(f, C);
                const auto p32 = gdn_nax_cpu::prepare<float>(f, C);
                const auto wy64 = gdn_nax_cpu::execute_prepared(f, p64);
                const auto wy32 = gdn_nax_cpu::execute_prepared(f, p32);
                const auto after64 = gdn_nax_cpu::transformed(continuation, C, &wy64.final_state);
                const auto after32 = gdn_nax_cpu::transformed(continuation, C, &wy32.final_state);
                if (cases++) std::cout << ',';
                const bool range = f.name == "extreme_range_prefix_underflow";
                const bool cancellation = f.name == "cancellation_repeated_key";
                std::cout << "{\"fixture\":" << quote(f.name) << ",\"rows\":" << f.rows
                          << ",\"chunk_size\":" << C << ",\"chunk_count\":" << p32.num_chunks
                          << ",\"prepared_scalar_stride\":" << p32.stride()
                          << ",\"continued_rows\":" << continuation.rows
                          << ",\"classification\":" << quote(range ? "F32_prefix_range_trap" :
                              cancellation ? "cancellation_fixture_gate_retained" : "ordinary")
                          << ",\"f64_equivalence\":";
                const auto c64 = emit_comparison(ref64, wy64, true);
                std::cout << ",\"f64_continued_equivalence\":";
                const auto a64 = emit_comparison(continued64, after64, true);
                const bool pass64 = c64.pass && a64.pass;
                f64_failures += !pass64;
                all64.merge(c64.aggregate); all64.merge(a64.aggregate);
                std::cout << ",\"f32_vs_serial_f64\":";
                const auto c32 = emit_comparison(ref64, wy32, false);
                std::cout << ",\"f32_continued_vs_serial_f64\":";
                const auto a32 = emit_comparison(continued64, after32, false);
                const bool pass32 = c32.pass && a32.pass;
                f32_failures += !pass32; non_range_f32_failures += !pass32 && !range;
                ordinary_f32_failures += !pass32 && !range && !cancellation;
                cancellation_failures += !pass32 && cancellation;
                all32.merge(c32.aggregate); all32.merge(a32.aggregate);
                std::cout << ",\"f32_association_vs_serial_f32\":";
                const auto association = emit_comparison(ref32, wy32, false);
                all32_association.merge(association.aggregate);
                std::cout << ",\"f32_continued_association_vs_serial_f32\":";
                all32_association.merge(emit_comparison(continued32, after32, false).aggregate);
                std::cout << ",\"serial_f32_vs_serial_f64\":";
                emit_comparison(ref64, ref32, false);
                const bool padding64 = padded_zero(p64), padding32 = padded_zero(p32);
                padding_failures += !padding64 || !padding32;
                std::cout << ",\"strict_checks\":{\"prepared_padding_f64_zero\":" << (padding64 ? "true" : "false")
                          << ",\"prepared_padding_f32_zero\":" << (padding32 ? "true" : "false");
                if (f.name == "prefix_underflow_surviving_update") {
                    const bool retained64 = exact_retained_update(wy64, f), retained32 = exact_retained_update(wy32, f);
                    strict_failures += !retained64 || !retained32;
                    std::cout << ",\"fresh_update_f64_exact\":" << (retained64 ? "true" : "false")
                              << ",\"fresh_update_f32_exact\":" << (retained32 ? "true" : "false");
                } else if (range) {
                    const bool source32 = exact_retained_range(ref32, f), retained64 = exact_retained_range(wy64, f);
                    const bool retained32 = exact_retained_range(wy32, f);
                    strict_failures += !source32 || !retained64;
                    expected_f32_range_losses += !retained32;
                    std::cout << ",\"serial_f32_range_retained\":" << (source32 ? "true" : "false")
                              << ",\"wy_f64_range_retained\":" << (retained64 ? "true" : "false")
                              << ",\"wy_f32_range_retained\":" << (retained32 ? "true" : "false")
                              << ",\"classification\":\"expected_F32_prefix_underflow_association_loss\"";
                }
                std::cout << "},\"f64_pass\":" << (pass64 ? "true" : "false")
                          << ",\"f32_quality_gate_pass\":" << (pass32 ? "true" : "false") << '}';
            }
        }
        const bool reference_pass = !f64_failures && !strict_failures && !padding_failures;
        const bool quality_pass = reference_pass && !non_range_f32_failures;
        std::cout << "],\"summary\":{\"case_count\":" << cases << ",\"fixture_count\":" << fixtures.size()
                  << ",\"f64_equivalence_failures\":" << f64_failures
                  << ",\"f32_quality_gate_failures\":" << f32_failures
                  << ",\"non_range_f32_quality_gate_failures\":" << non_range_f32_failures
                  << ",\"ordinary_f32_quality_gate_failures\":" << ordinary_f32_failures
                  << ",\"cancellation_f32_quality_gate_failures\":" << cancellation_failures
                  << ",\"strict_invariant_failures\":" << strict_failures
                  << ",\"prepared_padding_failures\":" << padding_failures
                  << ",\"expected_f32_range_loss_cases\":" << expected_f32_range_losses
                  << ",\"f64_all_fields_and_continuation\":";
        emit_metric(all64, true);
        std::cout << ",\"f32_all_fields_and_continuation\":";
        emit_metric(all32, false);
        std::cout << ",\"f32_association_all_fields_and_continuation\":";
        emit_metric(all32_association, false);
        std::cout << ",\"reference_test_pass\":" << (reference_pass ? "true" : "false")
                  << ",\"f32_quality_qualification_pass\":" << (quality_pass ? "true" : "false")
                  << ",\"pass\":" << (quality_pass ? "true" : "false") << "}}\n";
        // A cancellation failure remains a failed quality gate. Reporting it as
        // expected does not relax or bypass the established 1e-4 threshold.
        return reference_pass ? (quality_pass ? 0 : 2) : 1;
    } catch (const std::exception& e) {
        std::cerr << "CPU WY reference error: " << e.what() << '\n';
        return 1;
    }
}
