#pragma once
#include <cstdint>
#include <vector>

using limb_t = uint32_t;

// Multiply two large integers represented as base-2^30 limbs.
void host_multiply(const std::vector<limb_t> &A, const std::vector<limb_t> &B,
                   std::vector<limb_t> &C);