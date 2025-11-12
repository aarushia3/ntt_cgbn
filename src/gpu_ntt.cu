#include "modular_arith.cuh"  // <- your modular arithmetic definitions
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "ntt.cuh"

using namespace std;
using namespace gpuntt;

#define LOGN 2
#define BATCH 1

__host__ void gpu_ntt_forward(vector<uint32_t> &a) {
    cout << "Entering host side gpu_ntt_forward function" << endl;

    // need to convert to compatible data type
    std::vector<Data64> a64(a.begin(), a.end());

    size_t N = a.size();
    if (N == 0) return;

    // Primitive roots for one CRT prime
    // modulus, omega, psi
    NTTFactors factor(Modulus<Data64>(7681), (Data64)3383, (Data64)4298);
    NTTParameters parameters(LOGN, factor, ReductionPolynomial::X_N_minus);

    // CPU NTT
    NTTCPU<Data64> generator(parameters);
    vector<Data64> cpu_ntt_result;
    cpu_ntt_result = generator.ntt(a64);
    std::cout << "cpu_ntt_result = [ ";
    for (const auto& x : cpu_ntt_result)
        std::cout << x << " ";
    std::cout << "]" << std::endl;

    vector<Data64> cpu_intt_result;
    cpu_intt_result = generator.intt(cpu_ntt_result);
    std::cout << "cpu_intt_result = [ ";
    for (const auto& x : cpu_intt_result)
        std::cout << x << " ";
    std::cout << "]" << std::endl;
    
    // input copying to the device
    Data64* InOut_Datas;
    GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, parameters.n * sizeof(Data64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas, a64.data(), parameters.n * sizeof(Data64), cudaMemcpyHostToDevice));

    // Forward omega table allocation + generation
    Root<Data64>* Forward_Omega_Table_Device;
    GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<Data64>)));
    vector<Root<Data64>> forward_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.forward_root_of_unity_table);
    for (size_t i = 0; i < forward_omega_table.size(); i++) {
        printf("%zu: %llu\n", i, (unsigned long long)forward_omega_table[i]);
    }
    GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device, forward_omega_table.data(),
            parameters.root_of_unity_size * sizeof(Root<Data64>), cudaMemcpyHostToDevice));

    // GPU NTT inplace call
    Modulus<Data64>* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<Data64>)));
    Modulus<Data64> test_modulus_[1] = {parameters.modulus};
    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<Data64>), cudaMemcpyHostToDevice));

    ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = LOGN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};

    GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus,
                        cfg_ntt, BATCH, 1);

    // copying output to host
    Data64* Output_Host;
    Output_Host = (Data64*) malloc(BATCH * parameters.n * sizeof(Data64));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                       BATCH * parameters.n * sizeof(Data64),
                       cudaMemcpyDeviceToHost));

    
    for (long unsigned int i = 0; i < parameters.n; i++) {
        // if Data64 is integer-like (e.g., uint64_t, int64_t)
        printf("%llu ", (unsigned long long)Output_Host[i]);
    }

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for (int i = 0; i < BATCH; i++)
    {
        check = check_result(Output_Host + (i * parameters.n),
                                cpu_ntt_result.data(), parameters.n);

        if (!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (BATCH - 1)) && check)
        {
            cout << "All Correct for PerPolynomial NTT." << endl;
        }
    }

    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
    free(Output_Host);
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
    // std::vector<Root<Data32>> forward_omega_table(N);
    // for (size_t i = 0; i < N; i++) {
    //     forward_omega_table[i] = OPERATOR32::exp(root, i, mod_host); // root^i % p
    // }

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

// __host__ void gpu_pointwise_multiply(const std::vector<uint32_t> &a,
//                             const std::vector<uint32_t> &b,
//                             std::vector<uint32_t> &c, uint32_t p) {
//     size_t N = a.size();
//     if (N == 0 || b.size() != N) return;

// }