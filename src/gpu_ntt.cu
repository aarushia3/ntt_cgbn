#include "modular_arith.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "ntt.cuh"
#include "config.h"

using namespace std;
using namespace gpuntt;

NTTFactors<Data64> factors[4] = {
    {Modulus<Data64>(7681), 3383, 4298},
    {Modulus<Data64>(7681), 3383, 4298},
    {Modulus<Data64>(7681), 3383, 4298},
    {Modulus<Data64>(7681), 3383, 4298}
};

vector<uint32_t> moduli = {7681, 7681, 7681, 7681};

__host__ void gpu_ntt_forward(vector<uint32_t> &a, vector<vector<uint32_t>> &a_mod) {
    cout << "Entering host side gpu_ntt_forward function" << endl;

    // need to convert to compatible data type
    vector<Data64> a64(a.begin(), a.end());

    size_t N = a.size();
    if (N == 0) return;
    int logN = log2(static_cast<int>(N));

    for (int i = 0; i < NUM_MODULI; i ++ ) {
        NTTParameters parameters(logN, factors[i], ReductionPolynomial::X_N_minus); // N is the length of the array you are sending in

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
        for (size_t j = 0; j < forward_omega_table.size(); j++) {
            cout << "  Omega[" << j << "] = " << static_cast<unsigned long long>(forward_omega_table[j]) << endl;
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device, forward_omega_table.data(),
                parameters.root_of_unity_size * sizeof(Root<Data64>), cudaMemcpyHostToDevice));

        // GPU NTT inplace call
        Modulus<Data64>* test_modulus;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<Data64>)));
        Modulus<Data64> test_modulus_[1] = {parameters.modulus};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<Data64>), cudaMemcpyHostToDevice));

        ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};
    
        // launching kernel for gpu ntt in place
        GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus, cfg_ntt, BATCH, 1);

        // copying output to host
        Data64* Output_Host;
        Output_Host = (Data64*) malloc(parameters.n * sizeof(Data64));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                    parameters.n * sizeof(Data64),
                    cudaMemcpyDeviceToHost));

        cout << "[GPU] NTT output (device -> host): [ ";
        for (long unsigned int j = 0; j < parameters.n; j++) {
            cout << static_cast<unsigned long long>(Output_Host[j]) << " ";
        }
        cout << "]" << endl;

        // Comparing GPU NTT results and CPU NTT results
        bool check = true;
        for (int j = 0; j < BATCH; j++)
        {
            check = check_result(Output_Host + (j * parameters.n),
                                    cpu_ntt_result.data(), parameters.n);

            if (!check)
            {
                cout << "(in " << j << ". Poly.)" << endl;
                break;
            }

            if ((j == (BATCH - 1)) && check)
            {
                cout << "All Correct for PerPolynomial NTT." << endl;
            }
        }

        // copy Output_Host to a_mod[i]
        if (a_mod.empty()) a_mod.push_back(vector<uint32_t>());
        
        a_mod[i].clear();
        a_mod[i].reserve(parameters.n * 2);  // each data64 has two uint32_ts? i'm not sure how the datatypes will work
        
        for (size_t j = 0; j < parameters.n; j++) {
            uint32_t low  = static_cast<uint32_t>(Output_Host[j] & 0xFFFFFFFF);        // lower 32 bits
            // uint32_t high = static_cast<uint32_t>((Output_Host[j] >> 32) & 0xFFFFFFFF); // upper 32 bits
            a_mod[i].push_back(low);
            // a_mod[i].push_back(high);
        }

        cout << "[HOST] a_mod[" << i << "] = [ ";
        for (size_t k = 0; k < a_mod[i].size(); k++)
            cout << a_mod[i][k] << " ";
        cout << "]" << endl;

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
        free(Output_Host);
    }
}

__global__ void pointwise_mul_kernel(uint32_t* A, uint32_t* B, uint32_t* C,
                                     uint32_t modulus, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint64_t prod = static_cast<uint64_t>(A[idx]) * B[idx];
        C[idx] = static_cast<uint32_t>(prod % modulus);
    }
}

__host__ void gpu_pointwise_multiply(const vector<vector<uint32_t>>& A_mod, const vector<vector<uint32_t>>& B_mod, vector<vector<uint32_t>>& C_mod) {
    size_t N = A_mod[0].size();

    cout << "[HOST] Starting GPU pointwise multiplication" << endl;

    for (size_t m = 0; m < NUM_MODULI; ++m) {
        cout << "[HOST] A_mod for modulus " << m << " (mod = " << moduli[m] << "): [ ";
        for (size_t i = 0; i < N; ++i)
            cout << A_mod[m][i] << " ";
        cout << "]" << endl;

        cout << "[HOST] B_mod for modulus " << m << " (mod = " << moduli[m] << "): [ ";
        for (size_t i = 0; i < N; ++i)
            cout << B_mod[m][i] << " ";
        cout << "]" << endl;
    }

    C_mod.resize(NUM_MODULI, vector<uint32_t>(N));

    for (size_t m = 0; m < NUM_MODULI; ++m) {
        cout << "[HOST] Processing modulus " << m << " (mod = " << moduli[m] << ")" << endl;
        // construct device arrays
        const uint32_t* A_host = A_mod[m].data();
        const uint32_t* B_host = B_mod[m].data();

        uint32_t* C_host = C_mod[m].data();
        uint32_t modulus = moduli[m];

        uint32_t *A_dev, *B_dev, *C_dev;
        GPUNTT_CUDA_CHECK(cudaMalloc(&A_dev, N * sizeof(uint32_t)));
        GPUNTT_CUDA_CHECK(cudaMalloc(&B_dev, N * sizeof(uint32_t)));
        GPUNTT_CUDA_CHECK(cudaMalloc(&C_dev, N * sizeof(uint32_t)));

        GPUNTT_CUDA_CHECK(cudaMemcpy(A_dev, A_host, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaMemcpy(B_dev, B_host, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        pointwise_mul_kernel<<<blocks, threads>>>(A_dev, B_dev, C_dev, modulus, N);
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());

        GPUNTT_CUDA_CHECK(cudaMemcpy(C_host, C_dev, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        cout << "[HOST] Result for modulus " << m << ": [ ";
        for (size_t i = 0; i < N; ++i)
            cout << C_host[i] << " ";
        cout << "]" << endl;

        GPUNTT_CUDA_CHECK(cudaFree(A_dev));
        GPUNTT_CUDA_CHECK(cudaFree(B_dev));
        GPUNTT_CUDA_CHECK(cudaFree(C_dev));
    }
}

__host__ void gpu_ntt_inverse(vector<vector<uint32_t>> &c_mod, vector<vector<uint32_t>> &c_recovered) {
    cout << "Entering host side gpu_ntt_inverse function" << endl;

    for (int i = 0; i < NUM_MODULI; i ++ ) {
        // get the size of c_mod for logN in parameters
        size_t N = c_mod[i].size();
        if (N == 0) continue;
        int logN = log2(static_cast<int>(N));
        
        NTTParameters parameters(logN, factors[i], ReductionPolynomial::X_N_minus);

        vector<Data64> c64(c_mod[i].begin(), c_mod[i].end());
        
        // CPU INTT
        NTTCPU<Data64> generator(parameters);
        vector<Data64> cpu_intt_result = generator.intt(c64);
        cout << "[CPU] Inverse NTT result: [ ";
        for (const auto& x : cpu_intt_result)
            cout << x << " ";
        cout << "]" << endl;

        // input copying to the device
        Data64* InOut_Datas;
        GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, parameters.n * sizeof(Data64)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas, c64.data(), parameters.n * sizeof(Data64), cudaMemcpyHostToDevice));

        // Inverse omega table allocation + generation + copying to device
        Root<Data64>* Inverse_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<Data64>)));
        vector<Root<Data64>> inverse_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.inverse_root_of_unity_table);

        cout << "[GPU] Inverse omega table values:" << endl;
        for (size_t j = 0; j < inverse_omega_table.size(); j++) {
            cout << "  Omega[" << j << "] = " << static_cast<unsigned long long>(inverse_omega_table[j]) << endl;
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(Inverse_Omega_Table_Device, inverse_omega_table.data(),
            parameters.root_of_unity_size * sizeof(Root<Data64>), cudaMemcpyHostToDevice));

        // setting up modulus / mod_n_inverse to pass into ntt config
        Modulus<Data64>* test_modulus;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<Data64>)));
        Modulus<Data64> test_modulus_[1] = {parameters.modulus};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<Data64>), cudaMemcpyHostToDevice));

        // not sure what this does?
        Ninverse<Data64>* test_ninverse;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse<Data64>)));
        Ninverse<Data64> test_ninverse_[1] = {parameters.n_inv};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_, sizeof(Ninverse<Data64>), cudaMemcpyHostToDevice));

        ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = logN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        // output alloc + GPU INTT call
        Data64* Out_Datas;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Out_Datas, parameters.n * sizeof(Data64)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Out_Datas, c64.data(), parameters.n * sizeof(Data64), cudaMemcpyHostToDevice));
        GPU_INTT(InOut_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus, cfg_intt, BATCH, 1);

        // copying output to host
        Data64* Output_Host;

        Output_Host = (Data64*) malloc(parameters.n * sizeof(Data64));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Out_Datas, parameters.n * sizeof(Data64), cudaMemcpyDeviceToHost));

        cout << "[GPU] INTT output (device -> host): [ ";
        for (long unsigned int j = 0; j < parameters.n; j++) {
            cout << static_cast<unsigned long long>(Output_Host[j]) << " ";
        }
        cout << "]" << endl;

        // Comparing GPU NTT results and CPU NTT results
        bool check = true;
        for (int j = 0; j < BATCH; j++)
        {
            check = check_result(Output_Host + (j * parameters.n),
                                    cpu_intt_result.data(), parameters.n);

            if (!check)
            {
                cout << "(in " << j << ". Poly.)" << endl;
                break;
            }

            if ((j == (BATCH - 1)) && check)
            {
                cout << "All Correct for PerPolynomial INTT." << endl;
            }
        }

        // copy Output_Host to c_recovered[i]
        if (c_recovered.empty()) c_recovered.push_back(vector<uint32_t>());
        
        c_recovered[i].clear();
        c_recovered[i].reserve(parameters.n * 2);  // each data64 has two uint32_ts? i'm not sure how the datatypes will work
        
        for (size_t j = 0; j < parameters.n; j++) {
            uint32_t low  = static_cast<uint32_t>(Output_Host[j] & 0xFFFFFFFF);        // lower 32 bits
            // uint32_t high = static_cast<uint32_t>((Output_Host[j] >> 32) & 0xFFFFFFFF); // upper 32 bits
            c_recovered[i].push_back(low);
            // c_recovered[i].push_back(high);
        }

        cout << "[HOST] c_recovered[" << i << "] = [ ";
        for (size_t k = 0; k < c_recovered[i].size(); k++)
            cout << c_recovered[i][k] << " ";
        cout << "]" << endl;

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        free(Output_Host);
    }
}