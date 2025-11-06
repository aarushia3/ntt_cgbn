#include "modular_arith.cuh"  // <- your modular arithmetic definitions
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "ntt_merge/ntt.cuh"

__host__ void gpu_ntt_forward(std::vector<uint32_t> &a, uint32_t p) {
    std::cout << "Entering host side gpu_ntt_forward function" << std::endl;

    size_t N = a.size();
    if (N == 0) return;

    // Primitive roots for some CRT primes
    static const uint32_t CRT_PRIMES[] = {2013265921u, 1811939329u, 2113929217u};
    static const uint32_t CRT_ROOTS[] = {31u, 13u, 5u};
    uint32_t root = 0;

    for (size_t i = 0; i < sizeof(CRT_PRIMES)/sizeof(CRT_PRIMES[0]); i++) {
        if (CRT_PRIMES[i] == p) {
            root = CRT_ROOTS[i];
            break;
        }
    }
    if (root == 0) {
        std::cerr << "[gpu_ntt_forward] Error: Prime not recognized\n";
        return;
    }

    // Allocate input array on device
    uint32_t *d_a;
    cudaMalloc(&d_a, N * sizeof(uint32_t));
    cudaMemcpy(d_a, a.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Set up modulus on host and copy to device
    Modulus32 mod_host(p);
    Modulus32* d_mod;
    cudaMalloc(&d_mod, sizeof(Modulus32));
    cudaMemcpy(d_mod, &mod_host, sizeof(Modulus32), cudaMemcpyHostToDevice);

    // Generate forward root-of-unity table on host
    std::vector<Root<Data32>> forward_omega_table(N);
    for (size_t i = 0; i < N; i++) {
        forward_omega_table[i] = OPERATOR32::exp(root, i, mod_host); // root^i % p
    }

    std::cout << "Root: " << root << "\nOmega table: ";
    for (size_t i = 0; i < forward_omega_table.size(); i++)
        std::cout << forward_omega_table[i] << " ";
    std::cout << std::endl;

    // Copy omega table to device
    Root<Data32>* d_omega;
    cudaMalloc(&d_omega, forward_omega_table.size() * sizeof(Root<Data32>));
    cudaMemcpy(d_omega, forward_omega_table.data(),
               forward_omega_table.size() * sizeof(Root<Data32>),
               cudaMemcpyHostToDevice);

    // Configure GPU NTT
    gpuntt::ntt_rns_configuration<Data32> cfg_ntt;
    cfg_ntt.n_power = static_cast<int>(std::log2(N));
    cfg_ntt.ntt_type = gpuntt::FORWARD;
    cfg_ntt.ntt_layout = gpuntt::PerPolynomial;
    cfg_ntt.reduction_poly = gpuntt::ReductionPolynomial::X_N_minus;
    cfg_ntt.zero_padding = true;
    cfg_ntt.stream = 0;

    // Launch GPU NTT
    gpuntt::GPU_NTT_Inplace(d_a, d_omega, d_mod, cfg_ntt, 1, 1);

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(a.data(), d_a, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_omega);
    cudaFree(d_mod);
}

__host__ void gpu_pointwise_multiply(const std::vector<uint32_t> &a,
                            const std::vector<uint32_t> &b,
                            std::vector<uint32_t> &c, uint32_t p) {
    size_t N = a.size();
    if (N == 0 || b.size() != N) return;

}