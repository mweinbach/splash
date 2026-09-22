#pragma once

// Standalone CPU references. No Metal, MLX, model files, or production runtime.
// q/k: [rows, key_dim], v/beta/memory/delta/out: [rows, value_dim] except
// beta, which is one scalar per row. S: [value_dim, key_dim].

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gdn_chunk_cpu {

inline float from_bf16(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t to_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    // Round-to-nearest, ties-to-even, matching source BF16 inputs.
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

struct Fixture {
    std::string name;
    size_t rows = 0, key_dim = 128, value_dim = 128;
    std::vector<uint16_t> q, k, v, beta;
    std::vector<float> alpha, initial_state;

    void validate() const {
        if (!rows || !key_dim || !value_dim || q.size() != rows * key_dim ||
            k.size() != rows * key_dim || v.size() != rows * value_dim ||
            beta.size() != rows || alpha.size() != rows ||
            initial_state.size() != value_dim * key_dim)
            throw std::runtime_error("Invalid GDN CPU fixture dimensions");
    }
};

// memory_t = alpha_t * S_(t-1) * k_t (the decayed memory projection).
// history contains the complete post-update S_t for every token.
template <typename T> struct Trace {
    std::vector<T> memory, delta, out, history, final_state;
};

template <typename T> Trace<T> serial(const Fixture& f,
                                     const std::vector<T>* supplied_state = nullptr) {
    f.validate();
    const size_t K = f.key_dim, V = f.value_dim, M = K * V;
    Trace<T> r;
    r.memory.resize(f.rows * V);
    r.delta.resize(f.rows * V);
    r.out.resize(f.rows * V);
    r.history.resize(f.rows * M);
    std::vector<T> state(f.initial_state.begin(), f.initial_state.end());
    if (supplied_state) {
        if (supplied_state->size() != M) throw std::runtime_error("Invalid supplied state");
        state = *supplied_state;
    }
    for (size_t t = 0; t < f.rows; ++t) {
        const T a = T(f.alpha[t]), b = T(from_bf16(f.beta[t]));
        // Match the canonical source sequencing: decay every state element
        // before reducing the memory dot. This remains a scalar-dot reference;
        // a GPU SIMD reduction tree has its own additional association error.
        for (T& value : state) value *= a;
        for (size_t v = 0; v < V; ++v) {
            T memory = 0;
            for (size_t k = 0; k < K; ++k)
                memory += state[v * K + k] * T(from_bf16(f.k[t * K + k]));
            r.memory[t * V + v] = memory;
            r.delta[t * V + v] = b * (T(from_bf16(f.v[t * V + v])) - memory);
        }
        // Independent scalar recurrence, with no prefix or Gram matrices.
        for (size_t v = 0; v < V; ++v)
            for (size_t k = 0; k < K; ++k)
                state[v * K + k] += r.delta[t * V + v] * T(from_bf16(f.k[t * K + k]));
        std::copy(state.begin(), state.end(), r.history.begin() + t * M);
        for (size_t v = 0; v < V; ++v) {
            T out = 0;
            for (size_t k = 0; k < K; ++k)
                out += state[v * K + k] * T(from_bf16(f.q[t * K + k]));
            r.out[t * V + v] = out;
        }
    }
    r.final_state = std::move(state);
    return r;
}

template <typename T> Trace<T> chunked(const Fixture& f, size_t chunk_size,
                                      const std::vector<T>* supplied_state = nullptr) {
    f.validate();
    if (!chunk_size) throw std::runtime_error("Chunk size must be positive");
    const size_t K = f.key_dim, V = f.value_dim, M = K * V;
    Trace<T> r;
    r.memory.resize(f.rows * V);
    r.delta.resize(f.rows * V);
    r.out.resize(f.rows * V);
    r.history.resize(f.rows * M);
    std::vector<T> state(f.initial_state.begin(), f.initial_state.end());
    if (supplied_state) {
        if (supplied_state->size() != M) throw std::runtime_error("Invalid supplied state");
        state = *supplied_state;
    }
    for (size_t begin = 0; begin < f.rows; begin += chunk_size) {
        const size_t C = std::min(chunk_size, f.rows - begin);
        // Relative products are built directly: no A(t)/A(j) division.
        // Thus alpha=0 and underflowed prefixes retain all later contributions.
        std::vector<T> A(C), P(C * C, T(0)), gram(C * C, T(0)), qk(C * C, T(0));
        T prefix = 1;
        for (size_t t = 0; t < C; ++t) {
            prefix *= T(f.alpha[begin + t]);
            A[t] = prefix;
            P[t * C + t] = 1;
            T relative = 1;
            for (size_t j = t; j > 0; --j) {
                relative *= T(f.alpha[begin + j]);
                P[t * C + j - 1] = relative;
            }
            for (size_t j = 0; j <= t; ++j) {
                T kk = 0, q_dot_k = 0;
                for (size_t k = 0; k < K; ++k) {
                    const T kj = T(from_bf16(f.k[(begin + j) * K + k]));
                    kk += T(from_bf16(f.k[(begin + t) * K + k])) * kj;
                    q_dot_k += T(from_bf16(f.q[(begin + t) * K + k])) * kj;
                }
                gram[t * C + j] = kk;
                qk[t * C + j] = q_dot_k;
            }
        }
        std::vector<T> s0k(C * V), s0q(C * V);
        for (size_t t = 0; t < C; ++t)
            for (size_t v = 0; v < V; ++v) {
                T memory = 0, out = 0;
                for (size_t k = 0; k < K; ++k) {
                    memory += state[v * K + k] * T(from_bf16(f.k[(begin + t) * K + k]));
                    out += state[v * K + k] * T(from_bf16(f.q[(begin + t) * K + k]));
                }
                s0k[t * V + v] = memory;
                s0q[t * V + v] = out;
            }
        for (size_t t = 0; t < C; ++t) {
            const T b = T(from_bf16(f.beta[begin + t]));
            for (size_t v = 0; v < V; ++v) {
                const T base_memory = A[t] * s0k[t * V + v];
                T memory = base_memory;
                T delta = b * (T(from_bf16(f.v[(begin + t) * V + v])) - base_memory);
                for (size_t j = 0; j < t; ++j) {
                    const T coeff = P[t * C + j] * gram[t * C + j];
                    const T previous_delta = r.delta[(begin + j) * V + v];
                    memory += coeff * previous_delta;
                    // Direct triangular solve uses distributed beta, as in the
                    // proposed GPU algebra. This association differs in F32.
                    delta -= (b * coeff) * previous_delta;
                }
                r.memory[(begin + t) * V + v] = memory;
                r.delta[(begin + t) * V + v] = delta;
                T out = A[t] * s0q[t * V + v];
                for (size_t j = 0; j <= t; ++j)
                    out += (P[t * C + j] * qk[t * C + j]) * r.delta[(begin + j) * V + v];
                r.out[(begin + t) * V + v] = out;
            }
            // Every token state is reconstructed algebraically, independently
            // of the serial update. The last is carried into the next chunk.
            T* history = r.history.data() + (begin + t) * M;
            for (size_t v = 0; v < V; ++v)
                for (size_t k = 0; k < K; ++k) {
                    T value = A[t] * state[v * K + k];
                    for (size_t j = 0; j <= t; ++j)
                        value += (P[t * C + j] * r.delta[(begin + j) * V + v]) *
                            T(from_bf16(f.k[(begin + j) * K + k]));
                    history[v * K + k] = value;
                }
        }
        std::copy(r.history.begin() + (begin + C - 1) * M,
                  r.history.begin() + (begin + C) * M, state.begin());
    }
    r.final_state = std::move(state);
    return r;
}

// Fixture construction is implemented in cpu_oracle.cpp. Link with
// -DGDN_CPU_ORACLE_NO_MAIN to reuse it from a GPU oracle harness.
std::vector<Fixture> make_fixtures();
Fixture continuation_fixture(size_t rows = 19);
Fixture prefix_fixture(const Fixture& f, size_t rows);

} // namespace gdn_chunk_cpu
