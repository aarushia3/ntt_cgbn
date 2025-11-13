#pragma once
#include <cstdint>
#include <vector>
#include <iostream>

using namespace std;

void gpu_ntt_forward(std::vector<uint32_t> &a, vector<vector<uint32_t>> &a_mod);
void gpu_pointwise_multiply(const vector<vector<uint32_t>>& A_mod, const vector<vector<uint32_t>>& B_mod, vector<vector<uint32_t>>& C_mod);
void gpu_ntt_inverse(std::vector<uint32_t> &a, uint32_t p);
