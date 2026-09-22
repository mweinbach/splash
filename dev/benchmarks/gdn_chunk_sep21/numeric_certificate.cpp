#include "numeric_certificate.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace gdn_chunk_cpu {
namespace {
using R = long double;
constexpr R u32 = R(1) / R(16777216);

R gamma(size_t n, R u) {
    const R nu = R(n) * u;
    return nu < 1 ? nu / (1 - nu) : std::numeric_limits<R>::infinity();
}
R uref() { return std::ldexp(R(1), -std::numeric_limits<R>::digits); }
R gr(size_t n) { return gamma(n, uref()); }
R g32(size_t n) { return gamma(n, u32); }
R centered_reference_product_radius(size_t n, R computed) {
    const R g = gr(n);
    return g < 1 ? g / (1 - g) * std::abs(computed) : std::numeric_limits<R>::infinity();
}
// All bound arithmetic is nonnegative. Inflate each aggregate for its own
// reference evaluation; this fixed path budget is unrelated to observed errors.
R upper(R value, size_t steps = 64) {
    if (value == 0) return 0;
    return std::nextafter(value * (1 + gr(steps)), std::numeric_limits<R>::infinity());
}
R ratio(R numerator, R denominator) {
    return denominator > 0 ? numerator / denominator :
        (numerator == 0 ? R(0) : std::numeric_limits<R>::infinity());
}

void hazard(R value, DeltaCertificate& result) {
    if (value != 0 && (!std::isfinite(value) || std::abs(value) < R(std::numeric_limits<float>::min()) ||
                      std::abs(value) > R(std::numeric_limits<float>::max()))) ++result.range_hazards;
}

struct Projection { R value = 0, sum_abs = 0; };
Projection project(const std::vector<R>& state, size_t v, const Fixture& f, size_t t,
                   DeltaCertificate& result) {
    Projection p;
    for (size_t k = 0; k < f.key_dim; ++k) {
        const R term = state[v * f.key_dim + k] * R(from_bf16(f.k[t * f.key_dim + k]));
        hazard(term, result);
        p.value += term;
        p.sum_abs += std::abs(term);
    }
    p.sum_abs = upper(p.sum_abs, f.key_dim + 2);
    hazard(p.value, result);
    hazard(p.sum_abs, result); // Conservative possible reduction overflow check.
    return p;
}

R rhs_error(R a, R ea, const Projection& s, R es, R beta, R value, R u, size_t products) {
    const R em = upper(std::abs(a) * es + ea * (std::abs(s.value) + es) +
        u * (std::abs(a) + ea) * (std::abs(s.value) + es));
    return upper(beta * em + gamma(products, u) * beta *
        (std::abs(value) + std::abs(a * s.value) + em));
}

R coefficient_error(R p, R ep, R c, R ec, R beta, R u) {
    return upper(beta * (std::abs(p) * ec + ep * (std::abs(c) + ec)) +
                 gamma(2, u) * beta * (std::abs(p) + ep) * (std::abs(c) + ec));
}

struct Inverse {
    std::vector<R> x;
    R norm = 0, defect = 0;
    // B*X=I-F. B^-1=X*(I-F)^-1, with ||F||inf <= defect < 1.
    R apply_upper(const std::vector<R>& rhs, size_t t, size_t C) const {
        R sum = 0, peak = 0;
        for (size_t j = 0; j <= t; ++j) {
            sum += std::abs(x[t * C + j]) * rhs[j];
            peak = std::max(peak, rhs[j]);
        }
        return upper(sum + norm * defect / (1 - defect) * peak, 4 * C + 16);
    }
};

Inverse inverse_bound(const std::vector<R>& l, const std::vector<R>& el, size_t C) {
    Inverse r;
    r.x.assign(C * C, 0);
    for (size_t i = 0; i < C; ++i)
        for (size_t j = 0; j <= i; ++j) {
            R x = i == j ? 1 : 0;
            for (size_t k = j; k < i; ++k) x -= l[i * C + k] * r.x[k * C + j];
            r.x[i * C + j] = x;
        }
    for (size_t i = 0; i < C; ++i) {
        R row_norm = 0, row_defect = 0;
        for (size_t j = 0; j <= i; ++j) {
            row_norm += std::abs(r.x[i * C + j]);
            R bx = r.x[i * C + j], sum_abs = std::abs(bx), coefficient_eval = 0;
            for (size_t k = j; k < i; ++k) {
                const R term = l[i * C + k] * r.x[k * C + j];
                bx += term;
                sum_abs += std::abs(term);
                coefficient_eval += el[i * C + k] * std::abs(r.x[k * C + j]);
            }
            row_defect += std::abs((i == j ? R(1) : R(0)) - bx) +
                          coefficient_eval + gr(2 * C + 2) * sum_abs;
        }
        r.norm = std::max(r.norm, upper(row_norm, C + 2));
        r.defect = std::max(r.defect, upper(row_defect, 4 * C + 8));
    }
    return r;
}

void json_number(std::ostream& out, double x) {
    if (std::isfinite(x)) out << x; else out << "null";
}

} // namespace

double fp32_gamma(size_t n) { return double(g32(n)); }

DeltaCertificate certify_delta(const Fixture& f, const Trace<double>& truth,
                               const DeltaCandidateEvidence& candidate,
                               const DeltaCertificateOptions& opt) {
    f.validate();
    const size_t K = f.key_dim, V = f.value_dim, M = K * V, Cmax = opt.chunk_size;
    if (!Cmax || !opt.solve_roundings_per_term || !candidate.delta ||
        candidate.delta->size() != f.rows * V || truth.delta.size() != f.rows * V ||
        truth.memory.size() != f.rows * V || truth.history.size() != f.rows * M)
        throw std::runtime_error("Invalid delta certificate dimensions/options");
    const size_t num_chunks = (f.rows + Cmax - 1) / Cmax;
    if (candidate.captured_chunk_inputs && candidate.captured_chunk_inputs->size() != num_chunks * M)
        throw std::runtime_error("Invalid independently captured chunk inputs");
    if (candidate.reconstructed_history && candidate.reconstructed_history->size() != f.rows * M)
        throw std::runtime_error("Invalid reconstructed state history");
    DeltaCertificate result;
    result.rows = f.rows; result.key_dim = K; result.value_dim = V; result.chunk_size = Cmax;
    result.solve_roundings_per_term = opt.solve_roundings_per_term;
    result.history_proxy_absolute_uncertainty = opt.history_proxy_absolute_uncertainty;
    result.reference_evaluation_mantissa_bits = std::numeric_limits<R>::digits;
    result.elements.resize(f.rows * V);
    const bool proxy_known = std::isfinite(opt.history_proxy_absolute_uncertainty) &&
                             opt.history_proxy_absolute_uncertainty >= 0;
    if (candidate.captured_chunk_inputs) result.input_provenance = "independent_chunk_input_capture";
    else if (candidate.reconstructed_history) {
        result.input_provenance = proxy_known ? "history_proxy_with_supplied_uncertainty" : "history_proxy_uncertainty_unknown";
        result.incoming_state_identified = proxy_known || num_chunks == 1;
        if (!result.incoming_state_identified)
            result.limitations.push_back("Reconstructed history is not independently captured MMA carry; proxy uncertainty is unknown.");
    } else {
        result.input_provenance = "reference_state_only_actual_candidate_input_unknown";
        result.incoming_state_identified = num_chunks == 1;
        if (!result.incoming_state_identified)
            result.limitations.push_back("No candidate incoming state evidence after the first chunk.");
    }
    R error2 = 0, reference2 = 0, operand2 = 0;
    for (size_t t = 0; t < f.rows; ++t) {
        const R beta = from_bf16(f.beta[t]), alpha = f.alpha[t];
        if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)) result.assumptions_pass = false;
        R norm2 = 0;
        for (size_t k = 0; k < K; ++k) {
            const R x = from_bf16(f.k[t * K + k]);
            norm2 += x * x;
        }
        if (!std::isfinite(norm2) || std::sqrt(norm2) > R(opt.key_norm_limit)) result.assumptions_pass = false;
    }
    for (size_t begin = 0, chunk = 0; begin < f.rows; begin += Cmax, ++chunk) {
        const size_t C = std::min(Cmax, f.rows - begin);
        std::vector<R> input(M), reference_input(M);
        for (size_t i = 0; i < M; ++i) {
            reference_input[i] = begin ? R(truth.history[(begin - 1) * M + i]) : R(f.initial_state[i]);
            if (candidate.captured_chunk_inputs) input[i] = (*candidate.captured_chunk_inputs)[chunk * M + i];
            else if (begin && candidate.reconstructed_history) input[i] = (*candidate.reconstructed_history)[(begin - 1) * M + i];
            else input[i] = reference_input[i];
            if (!std::isfinite(input[i]) || !std::isfinite(reference_input[i])) result.assumptions_pass = false;
        }
        std::vector<R> A(C), L(C * C, 0), EL(C * C, 0), ELref(C * C, 0);
        R prefix = 1, minimum_prefix = std::numeric_limits<R>::infinity(), bnorm = 1;
        for (size_t t = 0; t < C; ++t) {
            prefix *= R(f.alpha[begin + t]); hazard(prefix, result); A[t] = prefix;
            if (prefix != 0) minimum_prefix = std::min(minimum_prefix, std::abs(prefix));
            R row_norm = 1;
            for (size_t j = 0; j < t; ++j) {
                R p = 1;
                for (size_t n = j + 1; n <= t; ++n) { p *= R(f.alpha[begin + n]); hazard(p, result); }
                R gram = 0, gram_abs = 0;
                for (size_t k = 0; k < K; ++k) {
                    const R term = R(from_bf16(f.k[(begin + t) * K + k])) *
                                   R(from_bf16(f.k[(begin + j) * K + k]));
                    hazard(term, result); gram += term; gram_abs += std::abs(term);
                }
                const R beta = from_bf16(f.beta[begin + t]);
                L[t * C + j] = beta * p * gram;
                hazard(gram, result); hazard(beta * gram, result); hazard(L[t * C + j], result);
                gram_abs = upper(gram_abs, K + 2);
                const R epref = centered_reference_product_radius(t - j, p), ecref = gr(K) * gram_abs;
                ELref[t * C + j] = coefficient_error(p, epref, gram, ecref, beta, uref());
                EL[t * C + j] = coefficient_error(p, g32(t - j) * (std::abs(p) + epref) + epref,
                    gram, g32(K) * gram_abs + ecref, beta, u32) + ELref[t * C + j];
                row_norm += std::abs(L[t * C + j]) + ELref[t * C + j];
            }
            bnorm = std::max(bnorm, upper(row_norm, 2 * C));
        }
        const Inverse inverse = inverse_bound(L, ELref, C);
        if (!(inverse.defect < 1)) {
            result.assumptions_pass = false;
            result.limitations.push_back("Reference inverse defect could not establish invertibility.");
        }
        DeltaChunkCondition condition;
        condition.begin = begin; condition.count = C;
        condition.lower_system_infinity_norm = double(bnorm);
        condition.inverse_defect_infinity_upper = double(inverse.defect);
        condition.inverse_infinity_norm_upper = double(inverse.norm / (1 - inverse.defect));
        condition.condition_infinity_upper = double(bnorm * inverse.norm / (1 - inverse.defect));
        condition.minimum_nonzero_prefix_magnitude = std::isfinite(minimum_prefix) ? double(minimum_prefix) : 0;
        R delta_peak = 0, rhs_peak = 0;
        for (size_t v = 0; v < V; ++v) {
            std::vector<R> local(C), state_rhs(C), residual_abs(C), residual_eval(C), recursive(C),
                           reference_solution(C), reference_allowance(C);
            for (size_t t = 0; t < C; ++t) {
                const size_t i = (begin + t) * V + v;
                auto& e = result.elements[i];
                const R beta = from_bf16(f.beta[begin + t]), value = from_bf16(f.v[i]);
                const Projection s = project(input, v, f, begin + t, result);
                const Projection sr = project(reference_input, v, f, begin + t, result);
                const R a = A[t], earef = centered_reference_product_radius(t + 1, a), esref = gr(K) * s.sum_abs;
                const R b = beta * (value - a * s.value);
                hazard(a * s.value, result); hazard(b, result);
                const R ebref = rhs_error(a, earef, s, esref, beta, value, uref(), 2);
                const R eb = rhs_error(a, g32(t + 1) * (std::abs(a) + earef) + earef, s,
                                      g32(K) * s.sum_abs + esref, beta, value, u32, 2) + ebref;
                R state_projection_error = 0;
                for (size_t k = 0; k < K; ++k) {
                    const R uncertainty = begin && !candidate.captured_chunk_inputs && proxy_known ?
                                          R(opt.history_proxy_absolute_uncertainty) : R(0);
                    state_projection_error += (std::abs(input[v * K + k] - reference_input[v * K + k]) + uncertainty) *
                                              std::abs(R(from_bf16(f.k[(begin + t) * K + k])));
                }
                state_rhs[t] = upper(beta * (std::abs(a) + earef) * state_projection_error, 2 * K + 8);
                const R d = (*candidate.delta)[i];
                if (!std::isfinite(d) || !std::isfinite(value) || !std::isfinite(truth.delta[i]) ||
                    !std::isfinite(truth.memory[i])) result.assumptions_pass = false;
                delta_peak = std::max(delta_peak, std::abs(d)); rhs_peak = std::max(rhs_peak, std::abs(b));
                R coefficient = 0, solve_magnitude = std::abs(b) + eb, residual = d - b;
                R residual_magnitude = std::abs(d) + std::abs(b), evaluation = ebref;
                R truth_solution = beta * (value - a * sr.value);
                for (size_t j = 0; j < t; ++j) {
                    const R previous = (*candidate.delta)[(begin + j) * V + v];
                    const R l = L[t * C + j], term = l * previous;
                    hazard(term, result);
                    coefficient += EL[t * C + j] * std::abs(previous);
                    solve_magnitude += (std::abs(l) + EL[t * C + j]) * std::abs(previous);
                    residual += term; residual_magnitude += std::abs(term);
                    evaluation += ELref[t * C + j] * std::abs(previous);
                    truth_solution -= l * reference_solution[j];
                }
                reference_solution[t] = truth_solution;
                const R solve_error = upper(g32(opt.solve_roundings_per_term * t) * solve_magnitude, 4 * C + 8);
                local[t] = upper(eb + coefficient + solve_error);
                residual_eval[t] = upper(evaluation + gr(2 * t + 2) * residual_magnitude, 4 * C + 8);
                residual_abs[t] = std::abs(residual);
                reference_allowance[t] = std::abs(truth_solution - R(truth.delta[i]));
                recursive[t] = local[t] + state_rhs[t];
                for (size_t j = 0; j < t; ++j)
                    recursive[t] += (std::abs(L[t * C + j]) + ELref[t * C + j]) * recursive[j];
                recursive[t] = upper(recursive[t], 4 * C + 8);
                e.absolute_error = double(std::abs(d - R(truth.delta[i])));
                e.cancellation_operand_scale = double(beta * (std::abs(value) + std::abs(R(truth.memory[i]))));
                e.rhs_rounding_bound = double(eb); e.incoming_state_rhs_bound = double(state_rhs[t]);
                e.coefficient_rounding_contribution = double(coefficient); e.solve_rounding_bound = double(solve_error);
                e.exact_system_residual = double(residual); e.residual_evaluation_bound = double(residual_eval[t]);
                e.local_residual_rounding_bound = double(local[t]);
                e.recursive_forward_bound = double(recursive[t]);
            }
            std::vector<R> forward_rhs(C), posterior_rhs(C), reference_eval_rhs(C);
            for (size_t t = 0; t < C; ++t) {
                forward_rhs[t] = local[t] + state_rhs[t];
                posterior_rhs[t] = residual_abs[t] + residual_eval[t] + state_rhs[t];
                const Projection sr = project(reference_input, v, f, begin + t, result);
                const R beta = from_bf16(f.beta[begin + t]), value = from_bf16(f.v[(begin + t) * V + v]);
                R mag = beta * (std::abs(value) + std::abs(A[t] * sr.value)) + std::abs(reference_solution[t]);
                reference_eval_rhs[t] = rhs_error(A[t], centered_reference_product_radius(t + 1, A[t]), sr,
                                                   gr(K) * sr.sum_abs, beta, value, uref(), 2);
                for (size_t j = 0; j < t; ++j) {
                    mag += std::abs(L[t * C + j] * reference_solution[j]);
                    reference_eval_rhs[t] += ELref[t * C + j] * std::abs(reference_solution[j]);
                }
                reference_eval_rhs[t] += gr(2 * t + 2) * mag;
            }
            for (size_t t = 0; t < C; ++t) {
                const size_t i = (begin + t) * V + v;
                auto& e = result.elements[i];
                const R target_allowance = reference_allowance[t] + inverse.apply_upper(reference_eval_rhs, t, C);
                e.reference_consistency_allowance = double(target_allowance);
                e.inverse_forward_bound = double(inverse.apply_upper(forward_rhs, t, C) + target_allowance);
                e.posterior_forward_bound = double(inverse.apply_upper(posterior_rhs, t, C) + target_allowance);
                e.recursive_forward_bound += double(target_allowance);
                const R rn = ratio(residual_abs[t], local[t] + residual_eval[t]);
                const R fn = ratio(R(e.absolute_error), R(e.inverse_forward_bound));
                result.maximum_residual_over_rounding_bound = std::max(result.maximum_residual_over_rounding_bound, double(rn));
                result.maximum_error_over_inverse_forward_bound = std::max(result.maximum_error_over_inverse_forward_bound, double(fn));
                result.maximum_error_over_recursive_forward_bound = std::max(result.maximum_error_over_recursive_forward_bound,
                    double(ratio(R(e.absolute_error), R(e.recursive_forward_bound))));
                result.rounding_consistent &= rn <= 1;
                result.forward_bound_pass &= fn <= 1;
                result.maximum_absolute_error = std::max(result.maximum_absolute_error, e.absolute_error);
                result.maximum_error_over_u_operand_scale = std::max(result.maximum_error_over_u_operand_scale,
                    double(ratio(R(e.absolute_error), u32 * R(e.cancellation_operand_scale))));
                result.zero_delta_with_nonzero_operands += truth.delta[i] == 0 && e.cancellation_operand_scale != 0;
                error2 += R(e.absolute_error) * R(e.absolute_error);
                reference2 += R(truth.delta[i]) * R(truth.delta[i]);
                operand2 += R(e.cancellation_operand_scale) * R(e.cancellation_operand_scale);
            }
        }
        condition.maximum_delta_over_rhs_peak = double(ratio(delta_peak, rhs_peak));
        result.chunks.push_back(condition);
    }
    result.rms_absolute_error = double(std::sqrt(error2 / R(f.rows * V)));
    result.relative_l2_error = double(std::sqrt(ratio(error2, reference2)));
    result.operand_condition_l2 = double(std::sqrt(ratio(operand2, reference2)));
    if (result.range_hazards) {
        result.assumptions_pass = false;
        result.limitations.push_back("Nonzero subnormal/overflow-risk intermediate violates the relative gamma-only model; no FTZ or underflow allowance was added.");
    }
    if (!result.assumptions_pass)
        result.limitations.push_back("Certificate requires finite source/candidate values, alpha/beta in [0,1], normalized bounded keys and normal representable products.");
    result.certificate_available = result.assumptions_pass && result.incoming_state_identified;
    return result;
}

std::string delta_certificate_json(const DeltaCertificate& r, bool include_elements) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\"schema\":\"gdn_delta_numeric_certificate_v1\",\"rows\":" << r.rows
        << ",\"K\":" << r.key_dim << ",\"V\":" << r.value_dim << ",\"chunk_size\":" << r.chunk_size
        << ",\"u_fp32\":" << double(u32) << ",\"gamma_K\":" << fp32_gamma(r.key_dim)
        << ",\"solve_roundings_per_term\":" << r.solve_roundings_per_term
        << ",\"reference_mantissa_bits\":" << r.reference_evaluation_mantissa_bits
        << ",\"assumptions_pass\":" << (r.assumptions_pass ? "true" : "false")
        << ",\"incoming_state_identified\":" << (r.incoming_state_identified ? "true" : "false")
        << ",\"certificate_available\":" << (r.certificate_available ? "true" : "false")
        << ",\"rounding_consistent\":" << (r.rounding_consistent ? "true" : "false")
        << ",\"forward_bound_pass\":" << (r.forward_bound_pass ? "true" : "false")
        << ",\"range_hazards\":" << r.range_hazards << ",\"input_provenance\":\"" << r.input_provenance << '"'
        << ",\"zero_delta_with_nonzero_operands\":" << r.zero_delta_with_nonzero_operands;
    out << ",\"history_proxy_absolute_uncertainty\":";
    json_number(out,r.history_proxy_absolute_uncertainty);
    const char* names[] = {"max_absolute_error", "rms_absolute_error", "relative_l2_error", "operand_condition_l2",
        "max_error_over_u_operand_scale", "max_residual_over_rounding_bound", "max_error_over_recursive_forward_bound",
        "max_error_over_inverse_forward_bound"};
    const double values[] = {r.maximum_absolute_error,r.rms_absolute_error,r.relative_l2_error,r.operand_condition_l2,
        r.maximum_error_over_u_operand_scale,r.maximum_residual_over_rounding_bound,
        r.maximum_error_over_recursive_forward_bound,r.maximum_error_over_inverse_forward_bound};
    for (size_t i = 0; i < 8; ++i) { out << ",\"" << names[i] << "\":"; json_number(out, values[i]); }
    out << ",\"limitations\":[";
    for (size_t i = 0; i < r.limitations.size(); ++i) { if(i) out << ','; out << '"' << r.limitations[i] << '"'; }
    out << "],\"chunks\":[";
    for (size_t i = 0; i < r.chunks.size(); ++i) {
        if(i) out << ',';
        const auto& c=r.chunks[i];
        out << "{\"begin\":" << c.begin << ",\"count\":" << c.count
            << ",\"B_inf\":" << c.lower_system_infinity_norm
            << ",\"inverse_inf_upper\":" << c.inverse_infinity_norm_upper
            << ",\"condition_inf_upper\":" << c.condition_infinity_upper
            << ",\"inverse_defect_inf_upper\":" << c.inverse_defect_infinity_upper
            << ",\"minimum_nonzero_prefix\":" << c.minimum_nonzero_prefix_magnitude
            << ",\"maximum_delta_over_rhs_peak\":";
        json_number(out,c.maximum_delta_over_rhs_peak); out << '}';
    }
    out << ']';
    if (include_elements) {
        out << ",\"elements\":[";
        for (size_t i = 0; i < r.elements.size(); ++i) {
            if(i) out << ',';
            const auto& e=r.elements[i];
            out << "{\"index\":" << i << ",\"abs_error\":" << e.absolute_error
                << ",\"operand_scale\":" << e.cancellation_operand_scale
                << ",\"rhs_rounding_bound\":" << e.rhs_rounding_bound
                << ",\"incoming_state_rhs_bound\":" << e.incoming_state_rhs_bound
                << ",\"coefficient_contribution\":" << e.coefficient_rounding_contribution
                << ",\"solve_rounding_bound\":" << e.solve_rounding_bound
                << ",\"residual\":" << e.exact_system_residual
                << ",\"residual_evaluation_bound\":" << e.residual_evaluation_bound
                << ",\"local_rounding_bound\":" << e.local_residual_rounding_bound
                << ",\"recursive_forward_bound\":" << e.recursive_forward_bound
                << ",\"inverse_forward_bound\":" << e.inverse_forward_bound
                << ",\"posterior_forward_bound\":" << e.posterior_forward_bound
                << ",\"reference_consistency_allowance\":" << e.reference_consistency_allowance << '}';
        }
        out << ']';
    }
    out << '}';
    return out.str();
}

} // namespace gdn_chunk_cpu

#ifndef GDN_NUMERIC_CERTIFICATE_NO_MAIN
int main() {
    using namespace gdn_chunk_cpu;
    const auto fixtures=make_fixtures();
    bool pass=true;
    for (const Fixture& f : fixtures) {
        const auto truth=serial<double>(f);
        if(f.name=="cancellation_repeated_key") {
            const auto native=serial<float>(f);
            long double error2=0,reference2=0,max_error=0;
            for(size_t i=0;i<truth.delta.size();++i) {
                const long double e=std::abs(static_cast<long double>(native.delta[i])-truth.delta[i]);
                error2+=e*e; reference2+=static_cast<long double>(truth.delta[i])*truth.delta[i]; max_error=std::max(max_error,e);
            }
            std::cout << std::setprecision(17) << "{\"kind\":\"native_cpu_serial_evidence\",\"fixture\":\"" << f.name
                      << "\",\"delta_relative_l2\":" << double(std::sqrt(error2/reference2))
                      << ",\"delta_max_abs\":" << double(max_error) << ",\"old_relative_guard\":0.0001,"
                         "\"old_relative_guard_pass\":false,\"threshold_changed\":false}\n";
        }
        for(size_t C : {size_t(16),size_t(32)}) {
            const auto candidate=chunked<float>(f,C);
            const size_t M=f.key_dim*f.value_dim,N=(f.rows+C-1)/C;
            std::vector<float> inputs(N*M);
            // CPU chunked() copies its final history verbatim into actual carry.
            // This identity is source-known for this CPU example only.
            for(size_t c=0;c<N;++c)
                for(size_t i=0;i<M;++i)
                    inputs[c*M+i]=c?candidate.history[(c*C-1)*M+i]:f.initial_state[i];
            DeltaCandidateEvidence evidence{&candidate.delta,&inputs,nullptr};
            DeltaCertificateOptions options; options.chunk_size=C;
            const auto certificate=certify_delta(f,truth,evidence,options);
            if(certificate.certificate_available) pass &= certificate.rounding_consistent && certificate.forward_bound_pass;
            std::cout << "{\"kind\":\"cpu_chunk_example\",\"fixture\":\"" << f.name
                      << "\",\"certificate\":" << delta_certificate_json(certificate) << "}\n";
        }
    }
    std::cout << "{\"kind\":\"summary\",\"available_certificates_pass\":" << (pass?"true":"false")
              << ",\"old_quality_thresholds_changed\":false}\n";
    return pass?0:1;
}
#endif
