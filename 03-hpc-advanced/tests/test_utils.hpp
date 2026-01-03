#pragma once

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace hpc::test {

// Tolerance comparison utilities
template <typename T>
bool almost_equal(T a, T b, T rel_tol = 1e-5, T abs_tol = 1e-6) {
    return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

template <typename T>
bool vectors_almost_equal(const std::vector<T>& a, const std::vector<T>& b,
                          T rel_tol = 1e-5, T abs_tol = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!almost_equal(a[i], b[i], rel_tol, abs_tol)) return false;
    }
    return true;
}

// Random data generators
template <typename T>
std::vector<T> random_vector(size_t n, T min_val = -1.0, T max_val = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(static_cast<float>(min_val),
                                                static_cast<float>(max_val));
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = static_cast<T>(dist(gen));
    }
    return result;
}

// RapidCheck generators
namespace gen {

inline rc::Gen<size_t> reasonable_size() {
    return rc::gen::inRange<size_t>(1, 1024 * 64);
}

inline rc::Gen<std::vector<float>> float_vector(size_t min_size = 1, size_t max_size = 1024) {
    return rc::gen::container<std::vector<float>>(
        rc::gen::inRange<size_t>(min_size, max_size),
        rc::gen::arbitrary<float>()
    );
}

} // namespace gen

} // namespace hpc::test
