#pragma once

#include <cstdint>
#include <vector>
#include "common/nttparameters.cuh"

// If ReductionPolynomial is small and used only here, forward-declare it
// Otherwise include the minimal header that defines it
// #include "common/nttparameters.cuh"

void gpu_ntt_forward(std::vector<uint32_t>& a, uint32_t p,
                     gpuntt::ReductionPolynomial poly = gpuntt::ReductionPolynomial::X_N_minus);

void gpu_pointwise_multiply(const std::vector<uint32_t>& a,
                            const std::vector<uint32_t>& b,
                            std::vector<uint32_t>& c, uint32_t p);

void gpu_ntt_inverse(std::vector<uint32_t>& a, uint32_t p);
