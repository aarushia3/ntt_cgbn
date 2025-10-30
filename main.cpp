// main.cpp
#include "include/gpu_ntt.h"
#include "include/multiply.h"
#include <cstdint>
#include <iostream>

int main() {
    std::vector<uint32_t> A = {1, 2, 3, 4};
    std::vector<uint32_t> B = {5, 6, 7, 8};
    std::vector<uint32_t> C;

    std::cout << "[Host] Multiplying small polynomials...\n";
    host_multiply(A, B, C);

    std::cout << "Result (host_multiply): ";
    for (auto x : C)
    std::cout << x << " ";
    std::cout << std::endl;

    std::cout << "Done.\n";

    return 0;
}
