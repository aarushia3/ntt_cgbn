#pragma once
#include <cstdint>
#include <vector>
#include <iostream>

void gpu_ntt_forward(std::vector<uint32_t> &a);
void gpu_pointwise_multiply(const std::vector<uint32_t> &a,
                            const std::vector<uint32_t> &b,
                            std::vector<uint32_t> &c, uint32_t p);
void gpu_ntt_inverse(std::vector<uint32_t> &a, uint32_t p);
