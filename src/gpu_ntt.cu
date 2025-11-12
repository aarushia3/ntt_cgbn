#include "modular_arith.cuh"  // <- your modular arithmetic definitions
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "ntt.cuh"

using namespace std;
using namespace gpuntt;

#define BATCH 1

__host__ void gpu_ntt_forward(vector<uint32_t> &a) {
    cout << "Entering host side gpu_ntt_forward function" << endl;

    // need to convert to compatible data type
    vector<Data64> a64(a.begin(), a.end());

    size_t N = a.size();
    if (N == 0) return;
    double logN = log2(static_cast<double>(N));

    // Primitive roots for one CRT prime
    NTTFactors factor(Modulus<Data64>(7681), (Data64)3383, (Data64)4298); // the order is the prime modulus (p), omega (root of unity), psi (inverse root of unity)
    NTTParameters parameters(logN, factor, ReductionPolynomial::X_N_minus); // N is the length of the array you are sending in

    // CPU NTT
    NTTCPU<Data64> generator(parameters);
    vector<Data64> cpu_ntt_result = generator.ntt(a64);
    cout << "[CPU] Forward NTT result: [ ";
    for (const auto& x : cpu_ntt_result)
        cout << x << " ";
    cout << "]" << endl;

    vector<Data64> cpu_intt_result = generator.intt(cpu_ntt_result);
    cout << "[CPU] Inverse NTT result: [ ";
    for (const auto& x : cpu_intt_result)
        cout << x << " ";
    cout << "]" << endl;
    
    // input copying to the device
    Data64* InOut_Datas;
    GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, parameters.n * sizeof(Data64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas, a64.data(), parameters.n * sizeof(Data64), cudaMemcpyHostToDevice));

    // Forward omega table allocation + generation + copying to device
    Root<Data64>* Forward_Omega_Table_Device;
    GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<Data64>)));
    vector<Root<Data64>> forward_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.forward_root_of_unity_table);

    cout << "[GPU] Forward omega table values:" << endl;
    for (size_t i = 0; i < forward_omega_table.size(); i++) {
        cout << "  Omega[" << i << "] = " << static_cast<unsigned long long>(forward_omega_table[i]) << endl;
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
    
    // launching kernel for gpu ntt in place
    GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus,
                        cfg_ntt, BATCH, 1);

    // copying output to host
    Data64* Output_Host;
    Output_Host = (Data64*) malloc(BATCH * parameters.n * sizeof(Data64));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                       BATCH * parameters.n * sizeof(Data64),
                       cudaMemcpyDeviceToHost));

    cout << "[GPU] NTT output (device -> host): [ ";
    for (long unsigned int i = 0; i < parameters.n; i++) {
        cout << static_cast<unsigned long long>(Output_Host[i]) << " ";
    }
    cout << "]" << endl;

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
}