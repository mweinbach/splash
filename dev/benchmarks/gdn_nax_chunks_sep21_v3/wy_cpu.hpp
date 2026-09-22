#pragma once

// Isolated scalar WY reference. All source inputs, fixture construction, and
// Trace layouts come from the independent serial CPU oracle. No Metal or
// production runtime headers are included here.
#include "frozen/cpu_oracle.hpp"

namespace gdn_nax_cpu {
using gdn_chunk_cpu::Fixture;
using gdn_chunk_cpu::Trace;
using gdn_chunk_cpu::from_bf16;

// Every chunk occupies one fixed stride, in this exact order:
// [W(T*K), U(T*V), E(T*K), score(T*T), prefix(T)].
// Matrices are row-major. T is chunk_size, including zero-padded final tails.
// With source K=V=128 this is [W(T*128), U(T*128), E(T*128),
// score(T*T), prefix(T)]. The per-chunk end row is the last *active* token.
template <typename T> struct Prepared {
    size_t rows = 0, chunk_size = 0, key_dim = 0, value_dim = 0, num_chunks = 0;
    std::vector<T> storage;

    size_t stride() const {
        return chunk_size * (2 * key_dim + value_dim) +
               chunk_size * chunk_size + chunk_size;
    }
    size_t active_rows(size_t chunk) const {
        if (chunk >= num_chunks) throw std::runtime_error("Invalid WY chunk index");
        return std::min(chunk_size, rows - chunk * chunk_size);
    }
    const T* w(size_t chunk) const { return storage.data() + chunk * stride(); }
    const T* u(size_t chunk) const { return w(chunk) + chunk_size * key_dim; }
    const T* e(size_t chunk) const { return u(chunk) + chunk_size * value_dim; }
    const T* score(size_t chunk) const { return e(chunk) + chunk_size * key_dim; }
    const T* prefix(size_t chunk) const { return score(chunk) + chunk_size * chunk_size; }
    T* w(size_t chunk) { return storage.data() + chunk * stride(); }
    T* u(size_t chunk) { return w(chunk) + chunk_size * key_dim; }
    T* e(size_t chunk) { return u(chunk) + chunk_size * value_dim; }
    T* score(size_t chunk) { return e(chunk) + chunk_size * key_dim; }
    T* prefix(size_t chunk) { return score(chunk) + chunk_size * chunk_size; }
};

// Compute P(i,j) directly. Never divide two prefixes: a zero or underflowed
// prefix must not erase a later rank-one update.
template <typename T> std::vector<T> relative_products(const Fixture& f,
                                                      size_t begin, size_t C) {
    std::vector<T> P(C * C, T(0));
    for (size_t i = 0; i < C; ++i) {
        P[i * C + i] = T(1);
        T relative = 1;
        for (size_t j = i; j > 0; --j) {
            relative *= T(f.alpha[begin + j]);
            P[i * C + j - 1] = relative;
        }
    }
    return P;
}

template <typename T> Prepared<T> prepare(const Fixture& f, size_t chunk_size = 16) {
    f.validate();
    if (!chunk_size) throw std::runtime_error("WY chunk size must be positive");
    const size_t K = f.key_dim, V = f.value_dim;
    Prepared<T> r;
    r.rows = f.rows; r.chunk_size = chunk_size; r.key_dim = K; r.value_dim = V;
    r.num_chunks = (f.rows + chunk_size - 1) / chunk_size;
    r.storage.assign(r.num_chunks * r.stride(), T(0));
    for (size_t chunk = 0; chunk < r.num_chunks; ++chunk) {
        const size_t begin = chunk * chunk_size, C = r.active_rows(chunk);
        const auto P = relative_products<T>(f, begin, C);
        std::vector<T> lower(C * C, T(0)), inverse(C * C, T(0));
        T prefix = 1;
        for (size_t i = 0; i < C; ++i) {
            prefix *= T(f.alpha[begin + i]);
            r.prefix(chunk)[i] = prefix;
            const T beta = T(from_bf16(f.beta[begin + i]));
            for (size_t j = 0; j <= i; ++j) {
                T gram = 0, qk = 0;
                for (size_t k = 0; k < K; ++k) {
                    const T kj = T(from_bf16(f.k[(begin + j) * K + k]));
                    gram += T(from_bf16(f.k[(begin + i) * K + k])) * kj;
                    qk += T(from_bf16(f.q[(begin + i) * K + k])) * kj;
                }
                if (j < i) lower[i * C + j] = (beta * P[i * C + j]) * gram;
                r.score(chunk)[i * chunk_size + j] = P[i * C + j] * qk;
            }
        }
        // Independently form F=(I+L)^-1 by solving each column. The main
        // recurrence below only sees W/U, never this triangular solve.
        for (size_t i = 0; i < C; ++i)
            for (size_t j = 0; j <= i; ++j) {
                T value = i == j ? T(1) : T(0);
                for (size_t n = j; n < i; ++n)
                    value -= lower[i * C + n] * inverse[n * C + j];
                inverse[i * C + j] = value;
            }
        for (size_t i = 0; i < C; ++i) {
            for (size_t k = 0; k < K; ++k) {
                T value = 0;
                for (size_t j = 0; j <= i; ++j) {
                    const T coefficient = (inverse[i * C + j] *
                        T(from_bf16(f.beta[begin + j]))) * r.prefix(chunk)[j];
                    value += coefficient * T(from_bf16(f.k[(begin + j) * K + k]));
                }
                r.w(chunk)[i * K + k] = value;
                r.e(chunk)[i * K + k] = P[(C - 1) * C + i] *
                    T(from_bf16(f.k[(begin + i) * K + k]));
            }
            for (size_t v = 0; v < V; ++v) {
                T value = 0;
                for (size_t j = 0; j <= i; ++j)
                    value += (inverse[i * C + j] * T(from_bf16(f.beta[begin + j]))) *
                        T(from_bf16(f.v[(begin + j) * V + v]));
                r.u(chunk)[i * V + v] = value;
            }
        }
    }
    return r;
}

template <typename T> Trace<T> execute_prepared(const Fixture& f, const Prepared<T>& p,
                                              const std::vector<T>* supplied_state = nullptr) {
    f.validate();
    const size_t K = f.key_dim, V = f.value_dim, M = K * V;
    if (!p.chunk_size || p.rows != f.rows || p.key_dim != K || p.value_dim != V ||
        p.num_chunks != (f.rows + p.chunk_size - 1) / p.chunk_size ||
        p.storage.size() != p.num_chunks * p.stride())
        throw std::runtime_error("Invalid prepared WY dimensions");
    Trace<T> r;
    r.memory.resize(f.rows * V); r.delta.resize(f.rows * V);
    r.out.resize(f.rows * V); r.history.resize(f.rows * M);
    // The source boundary is F32 even for F64 arithmetic, exactly as in serial.
    std::vector<T> state(f.initial_state.begin(), f.initial_state.end());
    if (supplied_state) {
        if (supplied_state->size() != M) throw std::runtime_error("Invalid supplied WY state");
        state = *supplied_state;
    }
    for (size_t chunk = 0; chunk < p.num_chunks; ++chunk) {
        const size_t begin = chunk * p.chunk_size, C = p.active_rows(chunk);
        const auto P = relative_products<T>(f, begin, C);
        const T* W = p.w(chunk); const T* U = p.u(chunk); const T* E = p.e(chunk);
        const T* score = p.score(chunk); const T* prefix = p.prefix(chunk);
        // D=U-W*S0^T, with D [C,V] and the source state [V,K].
        for (size_t i = 0; i < C; ++i)
            for (size_t v = 0; v < V; ++v) {
                T projected = 0;
                for (size_t k = 0; k < K; ++k) projected += W[i * K + k] * state[v * K + k];
                r.delta[(begin + i) * V + v] = U[i * V + v] - projected;
            }
        for (size_t i = 0; i < C; ++i) {
            for (size_t v = 0; v < V; ++v) {
                T memory_projection = 0, query_projection = 0;
                for (size_t k = 0; k < K; ++k) {
                    memory_projection += state[v * K + k] * T(from_bf16(f.k[(begin + i) * K + k]));
                    query_projection += state[v * K + k] * T(from_bf16(f.q[(begin + i) * K + k]));
                }
                T memory = prefix[i] * memory_projection;
                for (size_t j = 0; j < i; ++j) {
                    T gram = 0;
                    for (size_t k = 0; k < K; ++k)
                        gram += T(from_bf16(f.k[(begin + i) * K + k])) *
                                T(from_bf16(f.k[(begin + j) * K + k]));
                    memory += (P[i * C + j] * gram) * r.delta[(begin + j) * V + v];
                }
                r.memory[(begin + i) * V + v] = memory;
                T out = prefix[i] * query_projection;
                for (size_t j = 0; j <= i; ++j)
                    out += score[i * p.chunk_size + j] * r.delta[(begin + j) * V + v];
                r.out[(begin + i) * V + v] = out;
            }
            T* history = r.history.data() + (begin + i) * M;
            for (size_t v = 0; v < V; ++v)
                for (size_t k = 0; k < K; ++k) {
                    T value = prefix[i] * state[v * K + k];
                    for (size_t j = 0; j <= i; ++j) {
                        const T weighted_key = i + 1 == C ? E[j * K + k] :
                            P[i * C + j] * T(from_bf16(f.k[(begin + j) * K + k]));
                        value += r.delta[(begin + j) * V + v] * weighted_key;
                    }
                    history[v * K + k] = value;
                }
        }
        // S_end=prefix_end*S0+D^T*E. The exact carried value is also the last
        // reconstructed history row, including its floating point association.
        std::copy(r.history.begin() + (begin + C - 1) * M,
                  r.history.begin() + (begin + C) * M, state.begin());
    }
    r.final_state = std::move(state);
    return r;
}

template <typename T> Trace<T> transformed(const Fixture& f, size_t chunk_size = 16,
                                         const std::vector<T>* supplied_state = nullptr) {
    return execute_prepared(f, prepare<T>(f, chunk_size), supplied_state);
}
} // namespace gdn_nax_cpu
