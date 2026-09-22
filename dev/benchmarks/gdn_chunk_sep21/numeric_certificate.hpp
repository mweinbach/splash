#pragma once

#include "cpu_oracle.hpp"

#include <limits>
#include <string>
#include <vector>

namespace gdn_chunk_cpu {

// A separate numerical audit; this API does not modify any quality threshold.
struct DeltaCertificateOptions {
    size_t chunk_size = 16;
    // A nonfused multiply followed by subtract has two rounding steps.
    // One is permissible only when the compiled solve is known to use FMA.
    size_t solve_roundings_per_term = 2;
    // If boundary history is a proxy for actual MMA carry, a proven uniform
    // componentwise error bound may be supplied. NaN means unknown.
    double history_proxy_absolute_uncertainty = std::numeric_limits<double>::quiet_NaN();
    // A prequantization unit key rounded to BF16 RNE has norm <= 1+2^-8.
    double key_norm_limit = 1.0 + 1.0 / 256.0;
};

struct DeltaCandidateEvidence {
    const std::vector<float>* delta = nullptr; // [rows,V]
    // Preferred independent capture, before each chunk's actual projection:
    // [ceil(rows/chunk_size),V,K]. Do not label reconstructed history as capture.
    const std::vector<float>* captured_chunk_inputs = nullptr;
    const std::vector<float>* reconstructed_history = nullptr; // [rows,V,K]
};

struct DeltaElementCertificate {
    double absolute_error = 0;
    double cancellation_operand_scale = 0; // beta*(|v|+|serial memory|)
    double rhs_rounding_bound = 0;
    double incoming_state_rhs_bound = 0;
    double coefficient_rounding_contribution = 0;
    double solve_rounding_bound = 0;
    double exact_system_residual = 0;
    double residual_evaluation_bound = 0;
    double local_residual_rounding_bound = 0;
    double recursive_forward_bound = 0;
    double inverse_forward_bound = 0;
    double posterior_forward_bound = 0;
    double reference_consistency_allowance = 0;
};

struct DeltaChunkCondition {
    size_t begin = 0, count = 0;
    double lower_system_infinity_norm = 0;
    double inverse_infinity_norm_upper = 0;
    double condition_infinity_upper = 0;
    double inverse_defect_infinity_upper = 0;
    double minimum_nonzero_prefix_magnitude = 0;
    double maximum_delta_over_rhs_peak = 0;
};

struct DeltaCertificate {
    size_t rows = 0, key_dim = 0, value_dim = 0, chunk_size = 0;
    size_t solve_roundings_per_term = 2;
    double history_proxy_absolute_uncertainty = std::numeric_limits<double>::quiet_NaN();
    size_t reference_evaluation_mantissa_bits = 0;
    bool assumptions_pass = true;
    bool incoming_state_identified = true;
    bool certificate_available = true;
    bool rounding_consistent = true;
    bool forward_bound_pass = true;
    size_t range_hazards = 0, zero_delta_with_nonzero_operands = 0;
    std::string input_provenance;
    std::vector<std::string> limitations;
    std::vector<DeltaElementCertificate> elements;
    std::vector<DeltaChunkCondition> chunks;
    double maximum_absolute_error = 0;
    double rms_absolute_error = 0;
    double relative_l2_error = 0;
    double operand_condition_l2 = 0;
    double maximum_error_over_u_operand_scale = 0;
    double maximum_residual_over_rounding_bound = 0;
    double maximum_error_over_recursive_forward_bound = 0;
    double maximum_error_over_inverse_forward_bound = 0;
};

double fp32_gamma(size_t rounding_steps);

// truth must come from the independent F64 serial recurrence with identical
// quantized source inputs. The default implementation model is gamma_K dot,
// direct local products, two coefficient multiplies, and a direct row solve.
// A posteriori residuals are evaluated even when incoming state is a proxy;
// their provenance and availability are explicit in the result.
DeltaCertificate certify_delta(const Fixture& fixture, const Trace<double>& truth,
                               const DeltaCandidateEvidence& candidate,
                               const DeltaCertificateOptions& options = {});

std::string delta_certificate_json(const DeltaCertificate& result, bool include_elements = false);

} // namespace gdn_chunk_cpu
