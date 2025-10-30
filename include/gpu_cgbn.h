#pragma once
#include <cstdint>
#include <vector>

void gpu_crt_reconstruct(const std::vector<std::vector<uint32_t>> &C_mod,
                         std::vector<unsigned __int128> &C_big,
                         const uint32_t *MODULI, int num_moduli);

void gpu_carry_propagate(const std::vector<unsigned __int128> &C_big,
                         std::vector<uint32_t> &C, uint64_t base);
