#include "gpu_ntt.h"

#include "modular_arith.cuh"  // <- your modular arithmetic definitions
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "common/common.cuh"
#include "common/modular_arith.cuh"
#include "ntt_merge/ntt_cpu.cuh"
#include "ntt_merge/ntt.cuh"

using namespace gpuntt;

// int LOGN;
// int BATCH;
// using TestDataType = Data32; // or Data32
// Modulus<TestDataType> custom_modulus(2305843009213693951);

__host__ void gpu_ntt_forward(std::vector<uint32_t> &a, 
    uint32_t prime, 
    ReductionPolynomial poly) {
    std::cout << "Entering host side gpu_ntt_forward function" << std::endl;

    size_t N = a.size();
    if (N == 0) return;
    if ((N & (N - 1)) != 0) {
        throw std::invalid_argument("gpu_ntt_forward: input size N must be a power of two.");
    }

    int LOGN = static_cast<int>(std::log2(N));

    // build NTT parameter structure (Data32 = uint32_t typedef)
    NTTParameters<Data32> params(LOGN, poly);
    params.modulus = Modulus32(prime); // override the deafult modulus in GPU-NTT

    // 3) Build forward_root_of_unity_table that matches how GPU expects it:
    //    If you know the primitive root used by your GPU helper, build table
    //    using OPERATOR32::exp(root, k, params.modulus). Otherwise use
    //    params.forward_root_of_unity_table (if already filled).
    // Example: small primitive root mapping used in your project:
    uint32_t root = 0;
    // map known CRT primes to small primitive root (update as needed)
    if (prime == 2013265921u) root = 31u;
    else if (prime == 1811939329u) root = 13u;
    else if (prime == 2113929217u) root = 5u;
    else {
        std::cerr << "gpu_ntt_forward: unknown prime " << prime << ", cannot build root table\n";
        throw std::runtime_error("unknown modulus primitive root");
    }

    // fill parameters.forward_root_of_unity_table (length = params.root_of_unity_size)
    params.forward_root_of_unity_table.clear();
    params.forward_root_of_unity_table.reserve(params.root_of_unity_size);
    for (size_t k = 0; k < params.root_of_unity_size; ++k) {
        params.forward_root_of_unity_table.push_back(
            OPERATOR32::exp(static_cast<Data32>(root), static_cast<Data32>(k), params.modulus)
        );
    }

    // Convert to GPU-friendly Root<T> table (bitreversed internally if required)
    std::vector<Root<Data32>> forward_omega = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);

    // 4) allocate device memory and copy input + omega table + modulus if needed
    Data32* d_in = nullptr;
    GPUNTT_CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(Data32)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(d_in, a.data(), N * sizeof(Data32), cudaMemcpyHostToDevice));

    Root<Data32>* d_omega = nullptr;
    GPUNTT_CUDA_CHECK(cudaMalloc(&d_omega, forward_omega.size() * sizeof(Root<Data32>)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(d_omega, forward_omega.data(), forward_omega.size()*sizeof(Root<Data32>), cudaMemcpyHostToDevice));

    // some GPU wrapper signatures accept Modulus by value, some by pointer.
    // Here we'll upload a Modulus<Data32> to device if needed by the overload.
    Modulus<Data32>* d_mod = nullptr;
    GPUNTT_CUDA_CHECK(cudaMalloc(&d_mod, sizeof(Modulus<Data32>)));
    Modulus<Data32> mod_host = params.modulus;
    GPUNTT_CUDA_CHECK(cudaMemcpy(d_mod, &mod_host, sizeof(Modulus<Data32>), cudaMemcpyHostToDevice));

    // 5) build config
    ntt_rns_configuration<Data32> cfg = {
        .n_power = LOGN,
        .ntt_type = FORWARD,
        .ntt_layout = PerPolynomial,
        .reduction_poly = poly,
        .zero_padding = false,
        .mod_inverse = nullptr,
        .stream = 0
    };

    // 6) call GPU NTT (in-place)
    // note: depending on chosen overload this call may differ; there are multiple signatures
    // in your headers. Adjust the last args (batch and mod_count) as needed.
    gpuntt::GPU_NTT_Inplace<Data32>(d_in, d_omega, d_mod, cfg, 1, 1);

    GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());

    // 7) copy back
    a.assign(N, 0);
    GPUNTT_CUDA_CHECK(cudaMemcpy(a.data(), d_in, N * sizeof(Data32), cudaMemcpyDeviceToHost));

    // 8) cleanup
    GPUNTT_CUDA_CHECK(cudaFree(d_in));
    GPUNTT_CUDA_CHECK(cudaFree(d_omega));
    GPUNTT_CUDA_CHECK(cudaFree(d_mod));

    // Primitive roots for some CRT primes
    // static const uint32_t CRT_PRIMES[] = {2013265921u, 1811939329u, 2113929217u};
    // static const uint32_t CRT_ROOTS[] = {31u, 13u, 5u};
    // uint32_t root = 0;

    // for (size_t i = 0; i < sizeof(CRT_PRIMES)/sizeof(CRT_PRIMES[0]); i++) {
    //     if (CRT_PRIMES[i] == p) {
    //         root = CRT_ROOTS[i];
    //         break;
    //     }
    // }
    // if (root == 0) {
    //     std::cerr << "[gpu_ntt_forward] Error: Prime not recognized\n";
    //     return;
    // }

    // // Allocate input array on device
    // uint32_t *d_a;
    // cudaMalloc(&d_a, N * sizeof(uint32_t));
    // cudaMemcpy(d_a, a.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // // Set up modulus on host and copy to device
    // Modulus32 mod_host(p);
    // Modulus32* d_mod;
    // cudaMalloc(&d_mod, sizeof(Modulus32));
    // cudaMemcpy(d_mod, &mod_host, sizeof(Modulus32), cudaMemcpyHostToDevice);

    // // Generate forward root-of-unity table on host
    // // std::vector<Root<Data32>> forward_omega_table(N);
    // // for (size_t i = 0; i < N; i++) {
    // //     forward_omega_table[i] = OPERATOR32::exp(root, i, mod_host); // root^i % p
    // // }

    // std::vector<Root<Data32>> forward_omega_table =
    //         parameters.gpu_root_of_unity_table_generator(
    //             parameters.forward_root_of_unity_table);

    // std::cout << "Root: " << root << "\nOmega table: ";
    // for (size_t i = 0; i < forward_omega_table.size(); i++)
    //     std::cout << forward_omega_table[i] << " ";
    // std::cout << std::endl;

    // // Copy omega table to device
    // Root<Data32>* d_omega;
    // cudaMalloc(&d_omega, forward_omega_table.size() * sizeof(Root<Data32>));
    // cudaMemcpy(d_omega, forward_omega_table.data(),
    //            forward_omega_table.size() * sizeof(Root<Data32>),
    //            cudaMemcpyHostToDevice);

    // // Configure GPU NTT
    // gpuntt::ntt_rns_configuration<Data32> cfg_ntt;
    // cfg_ntt.n_power = static_cast<int>(std::log2(N));
    // cfg_ntt.ntt_type = gpuntt::FORWARD;
    // cfg_ntt.ntt_layout = gpuntt::PerPolynomial;
    // cfg_ntt.reduction_poly = gpuntt::ReductionPolynomial::X_N_minus;
    // cfg_ntt.zero_padding = true;
    // cfg_ntt.stream = 0;

    // // Launch GPU NTT
    // gpuntt::GPU_NTT_Inplace(d_a, d_omega, d_mod, cfg_ntt, 1, 1);

    // cudaDeviceSynchronize();

    // // Copy result back to host
    // cudaMemcpy(a.data(), d_a, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // // Cleanup
    // cudaFree(d_a);
    // cudaFree(d_omega);
    // cudaFree(d_mod);
}