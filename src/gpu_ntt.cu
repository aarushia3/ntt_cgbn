#include "gpu_ntt.h"
#include "modular_arith.cuh"
#include "zero_pad.h"
#include "carry_prop.h"
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ctime>

#include "ntt.cuh"
#include "config.h"
#include "ntt_limits.h"

using namespace std;
using namespace gpuntt;

#if LIMB_BITS == 64
    typedef Data64 TestDataType;
#else
    typedef Data32 TestDataType;
#endif

#if LIMB_BITS == 64 && defined(NATIVE_HOST_LIMBS)
    // 64-bit pipeline: 3×59-bit RNS, u160 carry
    vector<TestDataTypeUint> moduli = {
        0x400002600000001ULL,
        0x400004200000001ULL,
        0x400001100000001ULL,
    };
    vector<TestDataTypeUint> roots_of_unity_max = {
        273765203699653965ULL,
        26231613454922890ULL,
        144261359744151243ULL,
    };
#elif LIMB_BITS == 64
    // hybrid pipeline: 2×59-bit RNS, u128 carry
    vector<TestDataTypeUint> moduli = {0x400002600000001ULL, 0x400004200000001ULL};
    vector<TestDataTypeUint> roots_of_unity_max = {273765203699653965ULL, 26231613454922890ULL};
#else
    vector<TestDataTypeUint> moduli = {0x2d000001, 0x23800001, 0x26800001};
    vector<TestDataTypeUint> roots_of_unity_max = {663, 721, 19};
#endif

struct GPUTimer {
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    float toc(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

namespace {

struct StageProfiler {
    bool on = false;

    cudaEvent_t execute_start{};
    cudaEvent_t execute_stop{};
    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_stop_a{};
    cudaEvent_t h2d_stop_b{};
    cudaEvent_t fwd_start_a{};
    cudaEvent_t fwd_stop_a{};
    cudaEvent_t fwd_start_b{};
    cudaEvent_t fwd_stop_b{};

    unique_ptr<GPUTimer> mul_timer;
    unique_ptr<GPUTimer> intt_timer;
    unique_ptr<GPUTimer> crt_timer;
    unique_ptr<GPUTimer> carry_timer;
    unique_ptr<GPUTimer> d2h_timer;

    explicit StageProfiler(bool active) : on(active) {
        if (!on)
            return;
        cudaEventCreate(&execute_start);
        cudaEventCreate(&execute_stop);
        cudaEventCreate(&h2d_start);
        cudaEventCreate(&h2d_stop_a);
        cudaEventCreate(&h2d_stop_b);
        cudaEventCreate(&fwd_start_a);
        cudaEventCreate(&fwd_stop_a);
        cudaEventCreate(&fwd_start_b);
        cudaEventCreate(&fwd_stop_b);
        mul_timer = make_unique<GPUTimer>();
        intt_timer = make_unique<GPUTimer>();
        crt_timer = make_unique<GPUTimer>();
        carry_timer = make_unique<GPUTimer>();
        d2h_timer = make_unique<GPUTimer>();
    }

    ~StageProfiler() {
        if (!on)
            return;
        cudaEventDestroy(execute_start);
        cudaEventDestroy(execute_stop);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_stop_a);
        cudaEventDestroy(h2d_stop_b);
        cudaEventDestroy(fwd_start_a);
        cudaEventDestroy(fwd_stop_a);
        cudaEventDestroy(fwd_start_b);
        cudaEventDestroy(fwd_stop_b);
    }

    static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
        float ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    static float critical_path_ms(cudaEvent_t start,
                                cudaEvent_t stop_a,
                                cudaEvent_t stop_b) {
        float a = elapsed_ms(start, stop_a);
        float b = elapsed_ms(start, stop_b);
        return max(a, b);
    }

    void fill(NTTTiming& out) const {
        if (!on)
            return;

        out.h2d_ms = critical_path_ms(h2d_start, h2d_stop_a, h2d_stop_b);
        out.fwd_pad_ntt_a_ms = elapsed_ms(fwd_start_a, fwd_stop_a);
        out.fwd_pad_ntt_b_ms = elapsed_ms(fwd_start_b, fwd_stop_b);
        out.fwd_pad_ntt_ms = max(out.fwd_pad_ntt_a_ms, out.fwd_pad_ntt_b_ms);
        out.ingress_fwd_ms = max(
            elapsed_ms(h2d_start, fwd_stop_a),
            elapsed_ms(h2d_start, fwd_stop_b));
        out.total_ms = elapsed_ms(execute_start, execute_stop);
    }
};

} // namespace

// helper for modular multiplication; promotes to 64 / 128 bits to prevent overflow and then reduces mod
static TestDataType mod_mul(TestDataTypeUint a, TestDataTypeUint b, TestDataTypeUint mod) {
    if constexpr (sizeof(TestDataType) == 4) {
        return (TestDataType)((uint64_t)a * b % mod);
    } else {
        return (TestDataType)((__uint128_t)a * b % mod);
    }
}

// helper to generate new factors table compatible with given N
static array<NTTFactors<TestDataType>, NUM_MODULI> generate_factors_for_N(int logN) {
    array<NTTFactors<TestDataType>, NUM_MODULI> new_factors;
    for (int i = 0; i < NUM_MODULI; i++) {
        const int max_log = min(
            max_logN_for_prime(moduli[i]),
            max_root_logN(roots_of_unity_max[i], moduli[i]));

        if (logN > max_log) {
            cerr << "[NTT] logN=" << logN << " exceeds modulus " << i
                 << " (p=" << moduli[i] << ", max 2^" << max_log << ")\n";
            exit(1);
        }

        TestDataType root_max = roots_of_unity_max[i];
        TestDataType root_2_logN = root_max;
        for (int j = 0; j < (max_log - logN); j++) {
            root_2_logN = mod_mul(root_2_logN, root_2_logN, moduli[i]);
        }
        new_factors[i] = {moduli[i], root_2_logN, mod_mul(root_2_logN, root_2_logN, moduli[i])};
    }
    return new_factors;
}

template <typename T>
__device__ __forceinline__ void mul_wide(T a, T b, T &lo, T &hi)
{
    if constexpr (sizeof(T) == 4)
    {
        // 32-bit multiply -> 64-bit product
        uint64_t prod = (uint64_t)a * (uint64_t)b;
        lo = (T)prod;            // low 32 bits
        hi = (T)(prod >> 32);    // high 32 bits
    }
    else if constexpr (sizeof(T) == 8)
    {
        // 64-bit multiply -> 128-bit product
        lo = a * b;             // low 64 bits
        hi = __umul64hi(a, b);  // high 64 bits
    }
    else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                      "Unsupported TestDataTypeUint size.");
    }
}

__global__ void pointwise_mul_kernel(TestDataTypeUint* A,
                                     TestDataTypeUint* B,
                                     TestDataTypeUint* C,
                                     TestDataTypeUint modulus,
                                     size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        TestDataTypeUint lo, hi;
        mul_wide(A[idx], B[idx], lo, hi);

        if constexpr (sizeof(TestDataTypeUint) == 4) {
            uint64_t full = ((uint64_t)hi << 32) | lo;
            // TODO fix this to use barrett
            C[idx] = (TestDataTypeUint)(full % modulus);
        } else {
            unsigned __int128 full =
                ((unsigned __int128)hi << 64) | lo;
            // TODO this probably needs to be not a direct mod but barrett
            C[idx] = (TestDataTypeUint)(full % modulus);
        }
    }
}

NTTPrecomputed precompute_ntt(size_t N, PrecomputeTiming* timing_out) {
    using clock = chrono::high_resolution_clock;
    const auto wall_start = clock::now();

    auto elapsed_ms = [](const clock::time_point& t0,
                         const clock::time_point& t1) -> float {
        return chrono::duration<float, milli>(t1 - t0).count();
    };

    const bool profile = timing_out != nullptr;
    float params_ms = 0.0f;
    float twiddle_host_ms = 0.0f;

    ensure_ntt_size_supported(N);

    NTTPrecomputed pre;
    pre.N    = N;
    pre.logN = (int)log2((double)N);
    pre.gpu_uploaded = false;

    pre.params.resize(NUM_MODULI);
    pre.forward_omega_host.resize(NUM_MODULI);
    pre.inverse_omega_host.resize(NUM_MODULI);
    pre.forward_omega_dev.assign(NUM_MODULI, nullptr);
    pre.inverse_omega_dev.assign(NUM_MODULI, nullptr);
    pre.modulus_dev.assign(NUM_MODULI, nullptr);
    pre.ninv_dev.assign(NUM_MODULI, nullptr);

    auto t_factors_start = clock::now();
    auto factors = generate_factors_for_N(pre.logN);
    const auto t_factors_end = clock::now();

    for (int i = 0; i < NUM_MODULI; i++) {
        auto t_params_start = clock::now();
        pre.params[i] = NTTParameters<TestDataType>(
            pre.logN, factors[i], ReductionPolynomial::X_N_minus);
        const auto t_params_end = clock::now();
        if (profile)
            params_ms += elapsed_ms(t_params_start, t_params_end);

        auto &p = pre.params[i];

        auto t_twiddle_host_start = clock::now();
        pre.forward_omega_host[i] =
            p.gpu_root_of_unity_table_generator(p.forward_root_of_unity_table);
        pre.inverse_omega_host[i] =
            p.gpu_root_of_unity_table_generator(p.inverse_root_of_unity_table);
        const auto t_twiddle_host_end = clock::now();
        if (profile)
            twiddle_host_ms += elapsed_ms(t_twiddle_host_start, t_twiddle_host_end);
    }

    auto t_garner_start = clock::now();
    pre.garner = compute_garner_params(moduli);
    const auto wall_end = clock::now();

    if (profile) {
        timing_out->factors_ms = elapsed_ms(t_factors_start, t_factors_end);
        timing_out->params_ms = params_ms;
        timing_out->twiddle_host_ms = twiddle_host_ms;
        timing_out->garner_host_ms = elapsed_ms(t_garner_start, wall_end);
        timing_out->total_ms = elapsed_ms(wall_start, wall_end);
    }

    return pre;
}

void upload_ntt_precomputed(NTTPrecomputed& pre, SetupUploadTiming* timing_out) {
    using clock = chrono::high_resolution_clock;
    const auto wall_start = clock::now();

    auto elapsed_ms = [](const clock::time_point& t0,
                         const clock::time_point& t1) -> float {
        return chrono::duration<float, milli>(t1 - t0).count();
    };

    const bool profile = timing_out != nullptr;
    float twiddle_upload_ms = 0.0f;
    float mod_constants_ms = 0.0f;

    if (pre.gpu_uploaded)
        return;

    for (int i = 0; i < NUM_MODULI; i++) {
        auto &p = pre.params[i];
        const auto &fwd = pre.forward_omega_host[i];
        const auto &inv = pre.inverse_omega_host[i];

        auto t_twiddle_upload_start = clock::now();
        cudaMalloc(&pre.forward_omega_dev[i],
                   p.root_of_unity_size * sizeof(Root<TestDataType>));
        cudaMemcpy(pre.forward_omega_dev[i], fwd.data(),
                   p.root_of_unity_size * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice);

        cudaMalloc(&pre.inverse_omega_dev[i],
                   p.root_of_unity_size * sizeof(Root<TestDataType>));
        cudaMemcpy(pre.inverse_omega_dev[i], inv.data(),
                   p.root_of_unity_size * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice);
        const auto t_twiddle_upload_end = clock::now();
        if (profile)
            twiddle_upload_ms += elapsed_ms(t_twiddle_upload_start, t_twiddle_upload_end);

        auto t_mod_constants_start = clock::now();
        cudaMalloc(&pre.modulus_dev[i], sizeof(Modulus<TestDataType>));
        Modulus<TestDataType> mod_host[1] = {p.modulus};
        cudaMemcpy(pre.modulus_dev[i], mod_host,
                   sizeof(Modulus<TestDataType>), cudaMemcpyHostToDevice);

        cudaMalloc(&pre.ninv_dev[i], sizeof(Ninverse<TestDataType>));
        Ninverse<TestDataType> ninv_host[1] = {p.n_inv};
        cudaMemcpy(pre.ninv_dev[i], ninv_host,
                   sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice);
        const auto t_mod_constants_end = clock::now();
        if (profile)
            mod_constants_ms += elapsed_ms(t_mod_constants_start, t_mod_constants_end);
    }

    if (profile) {
        const auto t_sync_start = clock::now();
        cudaDeviceSynchronize();
        twiddle_upload_ms += elapsed_ms(t_sync_start, clock::now());
    } else {
        cudaDeviceSynchronize();
    }

    auto t_garner_upload_start = clock::now();
    upload_garner_params(pre.garner);
    const auto wall_end = clock::now();

    pre.gpu_uploaded = true;

    if (profile) {
        timing_out->twiddle_upload_ms = twiddle_upload_ms;
        timing_out->mod_constants_ms = mod_constants_ms;
        timing_out->garner_upload_ms = elapsed_ms(t_garner_upload_start, wall_end);
        timing_out->total_ms = elapsed_ms(wall_start, wall_end);
    }
}

// now allocate only owns the mutable buffers
NTTContext allocate_ntt_context(const NTTPrecomputed &pre, size_t L_A, size_t L_B) {
    if (!pre.gpu_uploaded) {
        cerr << "[NTT] allocate_ntt_context: call upload_ntt_precomputed first\n";
        exit(1);
    }

    NTTContext ctx;
    ctx.N    = pre.N;
    ctx.logN = pre.logN;
    ctx.L_A  = L_A;
    ctx.L_B  = L_B;

    ctx.a_dev.resize(NUM_MODULI);
    ctx.b_dev.resize(NUM_MODULI);
    ctx.c_dev.resize(NUM_MODULI);

    // borrow read-only pointers — no copy, no new allocation
    ctx.forward_omega_dev = pre.forward_omega_dev;
    ctx.inverse_omega_dev = pre.inverse_omega_dev;
    ctx.modulus_dev       = pre.modulus_dev;
    ctx.ninv_dev          = pre.ninv_dev;

    for (int i = 0; i < NUM_MODULI; i++) {
        const auto &p = pre.params[i];

        cudaMalloc(&ctx.a_dev[i], p.n * sizeof(TestDataType));
        cudaMalloc(&ctx.b_dev[i], p.n * sizeof(TestDataType));
        cudaMalloc(&ctx.c_dev[i], p.n * sizeof(TestDataType));
    }

    cudaMalloc(&ctx.a_raw_dev, L_A * sizeof(InputLimbType));
    cudaMalloc(&ctx.b_raw_dev, L_B * sizeof(InputLimbType));
#if defined(NATIVE_HOST_LIMBS)
    cudaMalloc(&ctx.d_C_lo, pre.N * sizeof(uint64_t));
    cudaMalloc(&ctx.d_C_mid, pre.N * sizeof(uint64_t));
    cudaMalloc(&ctx.d_C_hi32, pre.N * sizeof(uint32_t));
#else
    cudaMalloc(&ctx.d_C_hi, pre.N * sizeof(uint64_t));
    cudaMalloc(&ctx.d_C_lo, pre.N * sizeof(uint64_t));
#endif
    cudaMalloc(&ctx.d_out,  (pre.N + 1) * sizeof(OutputLimbType));
    size_t num_segs = (pre.N + CARRY_SEG - 1) / CARRY_SEG;
#if defined(NATIVE_HOST_LIMBS)
    cudaMalloc(&ctx.d_seg_carry_lo, num_segs * sizeof(uint64_t));
    cudaMalloc(&ctx.d_seg_carry_mid, num_segs * sizeof(uint64_t));
    cudaMalloc(&ctx.d_seg_carry_hi, num_segs * sizeof(uint32_t));
    cudaMalloc(&ctx.d_seg_carry_aux_lo, num_segs * sizeof(uint64_t));
    cudaMalloc(&ctx.d_seg_carry_aux_mid, num_segs * sizeof(uint64_t));
    cudaMalloc(&ctx.d_seg_carry_aux_hi, num_segs * sizeof(uint32_t));
#else
    cudaMalloc(&ctx.d_seg_carry, num_segs * sizeof(int64_t));
    cudaMalloc(&ctx.d_seg_carry_aux, num_segs * sizeof(int64_t));
#endif
    cudaMalloc(&ctx.d_carry_escape, sizeof(int));

    cudaStreamCreate(&ctx.stream_a);
    cudaStreamCreate(&ctx.stream_b);

    upload_residue_ptrs(ctx.c_dev);

    cudaDeviceSynchronize();
    return ctx;
}

void execute_ntt_multiply(
    NTTContext &ctx,
    const InputLimbType* a_pinned,
    const InputLimbType* b_pinned,
    vector<OutputLimbType> &C_out,
    NTTTiming* timing_out,
    vector<uint64_t>* crt_hi_out,
    vector<uint64_t>* crt_lo_out,
    vector<uint64_t>* crt_mid_out,
    vector<uint32_t>* crt_hi32_out)
{
    const bool profile = timing_out != nullptr;
#ifdef TIMING
    const bool write_timing_csv = true;
#else
    const bool write_timing_csv = false;
#endif
    StageProfiler prof(profile || write_timing_csv);
    NTTTiming timing{};

    if (prof.on)
        cudaEventRecord(prof.execute_start, ctx.stream_a);

    if (prof.on)
        cudaEventRecord(prof.h2d_start, ctx.stream_a);

    cudaMemcpyAsync(ctx.a_raw_dev, a_pinned,
                ctx.L_A * sizeof(InputLimbType),
                cudaMemcpyHostToDevice, ctx.stream_a);
    cudaMemcpyAsync(ctx.b_raw_dev, b_pinned,
                ctx.L_B * sizeof(InputLimbType),
                cudaMemcpyHostToDevice, ctx.stream_b);

    if (prof.on) {
        cudaEventRecord(prof.h2d_stop_a, ctx.stream_a);
        cudaEventRecord(prof.h2d_stop_b, ctx.stream_b);
        cudaEventRecord(prof.fwd_start_a, ctx.stream_a);
        cudaEventRecord(prof.fwd_start_b, ctx.stream_b);
    }
    
    #if DEBUG
    // verify inputs copied correctly
    cudaStreamSynchronize(ctx.stream_a);
    cudaStreamSynchronize(ctx.stream_b);
    vector<InputLimbType> chk_a(ctx.L_A), chk_b(ctx.L_B);
    cudaMemcpy(chk_a.data(), ctx.a_raw_dev, ctx.L_A*sizeof(InputLimbType), cudaMemcpyDeviceToHost);
    cudaMemcpy(chk_b.data(), ctx.b_raw_dev, ctx.L_B*sizeof(InputLimbType), cudaMemcpyDeviceToHost);
    printf("a_raw: "); for(auto x:chk_a) printf("%llu ",(unsigned long long)x); printf("\n");
    printf("b_raw: "); for(auto x:chk_b) printf("%llu ",(unsigned long long)x); printf("\n");
    #endif

    for (int i = 0; i < NUM_MODULI; i++) {
        ntt_rns_configuration<TestDataType> cfg_a = {
            .n_power = ctx.logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = ctx.stream_a
        };
        ntt_rns_configuration<TestDataType> cfg_b = {
            .n_power = ctx.logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = ctx.stream_b
        };
#if defined(NATIVE_HOST_LIMBS)
        zero_pad_gpu_u64(ctx.a_raw_dev, ctx.a_dev[i], ctx.L_A, ctx.N, moduli[i], ctx.stream_a);
        zero_pad_gpu_u64(ctx.b_raw_dev, ctx.b_dev[i], ctx.L_B, ctx.N, moduli[i], ctx.stream_b);
#else
        zero_pad_gpu(ctx.a_raw_dev, ctx.a_dev[i], ctx.L_A, ctx.N, ctx.stream_a);
        zero_pad_gpu(ctx.b_raw_dev, ctx.b_dev[i], ctx.L_B, ctx.N, ctx.stream_b);
#endif

        #if DEBUG
        // verify zero pad
        cudaStreamSynchronize(ctx.stream_a);
        vector<TestDataTypeUint> zp(ctx.N);
        cudaMemcpy(zp.data(), ctx.a_dev[i], ctx.N*sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);
        printf("a_dev[%d] zero_padded: ", i);
        for(int k=0;k<8;k++) printf("%u ",zp[k]); printf("\n");

        cudaStreamSynchronize(ctx.stream_b);
        cudaMemcpy(zp.data(), ctx.b_dev[i], ctx.N*sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);
        printf("b_dev[%d] zero_padded: ", i);
        for(int k=0;k<8;k++) printf("%u ",zp[k]); printf("\n");
        #endif

        #if DEBUG
        // compute CPU ntt
        int logN = log2(static_cast<int>(ctx.N));

        auto factors = generate_factors_for_N(logN);
        NTTParameters<TestDataType> parameters(
            logN,
            factors[i],
            ReductionPolynomial::X_N_minus
        );

        vector<TestDataType> host_a(ctx.N);
        vector<TestDataType> host_b(ctx.N);

        cudaMemcpy(
            host_a.data(),
            ctx.a_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            host_b.data(),
            ctx.b_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );

        NTTCPU<TestDataType> generatora(parameters);
        vector<TestDataType> cpu_ntt_result_a = generatora.ntt(host_a);

        NTTCPU<TestDataType> generatorb(parameters);
        vector<TestDataType> cpu_ntt_result_b = generatorb.ntt(host_b);

        cout << i << " :[CPU] Forward NTT result for a_dev: [ ";
        for (const auto& x : cpu_ntt_result_a)
            cout << x << " ";
        cout << "]" << endl;

        cout << i << " :[CPU] Forward NTT result for b_dev: [ ";
        for (const auto& x : cpu_ntt_result_b)
            cout << x << " ";
        cout << "]" << endl;
        #endif

        GPU_NTT_Inplace(ctx.a_dev[i], ctx.forward_omega_dev[i],
                        ctx.modulus_dev[i], cfg_a, BATCH, 1);

        GPU_NTT_Inplace(ctx.b_dev[i], ctx.forward_omega_dev[i],
                        ctx.modulus_dev[i], cfg_b, BATCH, 1);

        #if DEBUG
        vector<TestDataType> gpu_ntt_a(ctx.N);
        vector<TestDataType> gpu_ntt_b(ctx.N);

        cudaMemcpy(
            gpu_ntt_a.data(),
            ctx.a_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            gpu_ntt_b.data(),
            ctx.b_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );

        bool match_a = true;
        bool match_b = true;

        for (size_t j = 0; j < ctx.N; j++) {
            if (gpu_ntt_a[j] != cpu_ntt_result_a[j]) {
                match_a = false;
                printf("Mismatch A modulus %d index %zu : GPU=%llu CPU=%llu\n",
                    i,
                    j,
                    (unsigned long long)gpu_ntt_a[j],
                    (unsigned long long)cpu_ntt_result_a[j]);
                break;
            }
        }

        for (size_t j = 0; j < ctx.N; j++) {
            if (gpu_ntt_b[j] != cpu_ntt_result_b[j]) {
                match_b = false;
                printf("Mismatch B modulus %d index %zu : GPU=%llu CPU=%llu\n",
                    i,
                    j,
                    (unsigned long long)gpu_ntt_b[j],
                    (unsigned long long)cpu_ntt_result_b[j]);
                break;
            }
        }

        if (match_a)
            printf("[DEBUG] modulus %d : A NTT matched\n", i);

        if (match_b)
            printf("[DEBUG] modulus %d : B NTT matched\n", i);

        printf("midpoint value = %llu\n",
            (unsigned long long)
            gpu_ntt_a[gpu_ntt_a.size()/2]);

        printf("expected -1 mod p = %llu\n",
            (unsigned long long)
            (moduli[i] - 1));
        
        bool found_minus_one = false;

        for (size_t j = 0; j < gpu_ntt_a.size(); j++) {
            if (gpu_ntt_a[j] == moduli[i] - 1) {
                printf("-1 mod p found at index %zu\n", j);
                found_minus_one = true;
            }
        }

        if (!found_minus_one)
            printf("-1 mod p not found in transform output\n");
        #endif
    }

    cudaStreamSynchronize(ctx.stream_a);
    cudaStreamSynchronize(ctx.stream_b);

    if (prof.on) {
        cudaEventRecord(prof.fwd_stop_a, ctx.stream_a);
        cudaEventRecord(prof.fwd_stop_b, ctx.stream_b);
    }

    if (prof.on)
        prof.mul_timer->tic(ctx.stream_a);

    for (int i = 0; i < NUM_MODULI; i++) {
        int threads = 256;
        int blocks = (ctx.N + threads - 1) / threads;
        pointwise_mul_kernel<<<blocks, threads, 0, ctx.stream_a>>>(
            ctx.a_dev[i], ctx.b_dev[i], ctx.c_dev[i], moduli[i], ctx.N);
    }
    if (prof.on)
        timing.pointwise_mul_ms = prof.mul_timer->toc(ctx.stream_a);
    
    #if DEBUG
    // verify pointwise mul by printing c_dev
    cudaStreamSynchronize(ctx.stream_a);
    for (int i = 0; i < NUM_MODULI; i++) {
        vector<TestDataType> freq_domain_product(ctx.N);
        cudaMemcpy(
            freq_domain_product.data(),
            ctx.c_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );
        cout << "[GPU] Pointwise product for modulus " << i << ": [ ";
        for (long unsigned int j = 0; j < ctx.N; j++) {
            cout << static_cast<unsigned long long>(freq_domain_product[j]) << " ";
        }
        cout << "]" << endl;
    }
    #endif

    if (prof.on)
        prof.intt_timer->tic(ctx.stream_a);

    // inverse
    for (int i = 0; i < NUM_MODULI; i++) {
        ntt_rns_configuration<TestDataType> cfg_inv = {
            .n_power = ctx.logN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = ctx.ninv_dev[i],
            .stream = ctx.stream_a
        };

        #if DEBUG
        // pull pointwise multiplication result back before inverse transform
        vector<TestDataType> freq_domain_product(ctx.N);

        cudaMemcpy(
            freq_domain_product.data(),
            ctx.c_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );

        int logN = ctx.logN;

        auto factors = generate_factors_for_N(logN);
        NTTParameters<TestDataType> parameters(
            logN,
            factors[i],
            ReductionPolynomial::X_N_minus
        );

        NTTCPU<TestDataType> cpu_generator(parameters);

        vector<TestDataType> cpu_intt_result =
            cpu_generator.intt(freq_domain_product);

        cout << "[GPU] INTT output: [ ";
        for (long unsigned int j = 0; j < parameters.n; j++) {
            cout << static_cast<unsigned long long>(cpu_intt_result[j]) << " ";
        }
        cout << "]" << endl;
        #endif

        GPU_INTT(ctx.c_dev[i],
                 ctx.c_dev[i],
                 ctx.inverse_omega_dev[i],
                 ctx.modulus_dev[i],
                 cfg_inv,
                 BATCH, 1);

        #if DEBUG
        vector<TestDataType> gpu_intt_result(ctx.N);

        cudaMemcpy(
            gpu_intt_result.data(),
            ctx.c_dev[i],
            ctx.N * sizeof(TestDataType),
            cudaMemcpyDeviceToHost
        );

        bool intt_match = true;

        for (size_t j = 0; j < ctx.N; j++) {
            if (gpu_intt_result[j] != cpu_intt_result[j]) {
                intt_match = false;

                printf(
                    "INTT mismatch modulus %d index %zu : GPU=%llu CPU=%llu\n",
                    i,
                    j,
                    (unsigned long long)gpu_intt_result[j],
                    (unsigned long long)cpu_intt_result[j]
                );

                break;
            }
        }

        if (intt_match) {
            printf("[DEBUG] modulus %d : INTT matched\n", i);
        }
        #endif
    }

    cudaStreamSynchronize(ctx.stream_a);

    if (prof.on)
        timing.intt_ms = prof.intt_timer->toc(ctx.stream_a);

    if (prof.on)
        prof.crt_timer->tic(ctx.stream_a);

    #if DEBUG
    for (int mod = 0; mod < NUM_MODULI; mod++) {
        vector<TestDataTypeUint> tmp(ctx.N);

        cudaMemcpy(
            tmp.data(),
            ctx.c_dev[mod],
            ctx.N*sizeof(TestDataTypeUint),
            cudaMemcpyDeviceToHost
        );

        printf("Residues mod %d:\n", mod);

        for (int k=0;k<8;k++)
            printf("%llu ", (unsigned long long)tmp[k]);

        printf("\n");
    }
    #endif

    // ctx.c_dev[i] holds INTT results — pass directly to CRT, no host round-trip
#if defined(NATIVE_HOST_LIMBS)
    crt_combine_gpu_u160(ctx.d_C_lo, ctx.d_C_mid, ctx.d_C_hi32, ctx.N);
#else
    crt_combine_gpu(ctx.d_C_hi, ctx.d_C_lo, ctx.N);
#endif

    #if DEBUG && !defined(NATIVE_HOST_LIMBS)
    cudaDeviceSynchronize();
    vector<uint64_t> chi(8), clo(8);
    cudaMemcpy(chi.data(), ctx.d_C_hi, 8*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(clo.data(), ctx.d_C_lo, 8*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (int k = 0; k < 8; k++) {
        unsigned __int128 x =
            ((unsigned __int128)chi[k] << 64) |
            clo[k];

        unsigned long long hi =
            (unsigned long long)(x >> 64);

        unsigned long long lo =
            (unsigned long long)x;

        printf("CRT[%d] = hi=%llu lo=%llu\n",
            k, hi, lo);
    }
    #endif

    if (prof.on)
        timing.crt_ms = prof.crt_timer->toc(ctx.stream_a);

#if defined(NATIVE_HOST_LIMBS)
    if (crt_lo_out && crt_mid_out && crt_hi32_out) {
        crt_lo_out->resize(ctx.N);
        crt_mid_out->resize(ctx.N);
        crt_hi32_out->resize(ctx.N);
        cudaMemcpy(crt_lo_out->data(), ctx.d_C_lo, ctx.N * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(crt_mid_out->data(), ctx.d_C_mid, ctx.N * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(crt_hi32_out->data(), ctx.d_C_hi32, ctx.N * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        if (prof.on)
            cudaEventRecord(prof.execute_stop, 0);

        if (prof.on)
            prof.fill(timing);

        if (timing_out)
            *timing_out = timing;
        return;
    }
#else
    if (crt_hi_out && crt_lo_out) {
        crt_hi_out->resize(ctx.N);
        crt_lo_out->resize(ctx.N);
        cudaMemcpy(crt_hi_out->data(), ctx.d_C_hi, ctx.N * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(crt_lo_out->data(), ctx.d_C_lo, ctx.N * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);

        if (prof.on)
            cudaEventRecord(prof.execute_stop, 0);

        if (prof.on)
            prof.fill(timing);

        if (timing_out)
            *timing_out = timing;
        return;
    }
#endif

    if (prof.on)
        prof.carry_timer->tic(ctx.stream_a);

    size_t num_segs = (ctx.N + CARRY_SEG - 1) / CARRY_SEG;

#if defined(NATIVE_HOST_LIMBS)
    uint64_t* carry_in_lo  = ctx.d_seg_carry_lo;
    uint64_t* carry_in_mid = ctx.d_seg_carry_mid;
    uint32_t* carry_in_hi  = ctx.d_seg_carry_hi;
    uint64_t* carry_out_lo  = ctx.d_seg_carry_aux_lo;
    uint64_t* carry_out_mid = ctx.d_seg_carry_aux_mid;
    uint32_t* carry_out_hi  = ctx.d_seg_carry_aux_hi;
#else
    int64_t* carry_in  = ctx.d_seg_carry;
    int64_t* carry_out = ctx.d_seg_carry_aux;
#endif

#if defined(NATIVE_HOST_LIMBS)
    carry_intra_segment_kernel_u160<<<num_segs, 1, 0, ctx.stream_a>>>(
        ctx.d_C_lo, ctx.d_C_mid, ctx.d_C_hi32, ctx.d_out,
        carry_in_lo, carry_in_mid, carry_in_hi, ctx.N);
#else
    carry_intra_segment_kernel<<<num_segs, 1, 0, ctx.stream_a>>>(
        ctx.d_C_hi, ctx.d_C_lo, ctx.d_out, carry_in, ctx.N);
#endif

#if defined(NATIVE_HOST_LIMBS)
    carry_inter_segment_kernel_u160<<<1, 1, 0, ctx.stream_a>>>(
        carry_in_lo, carry_in_mid, carry_in_hi, num_segs);
#else
    carry_inter_segment_kernel<<<1, 1, 0, ctx.stream_a>>>(
        carry_in, num_segs);
#endif

    for (;;) {
        cudaMemsetAsync(ctx.d_carry_escape, 0, sizeof(int), ctx.stream_a);
#if defined(NATIVE_HOST_LIMBS)
        cudaMemsetAsync(carry_out_lo, 0, num_segs * sizeof(uint64_t), ctx.stream_a);
        cudaMemsetAsync(carry_out_mid, 0, num_segs * sizeof(uint64_t), ctx.stream_a);
        cudaMemsetAsync(carry_out_hi, 0, num_segs * sizeof(uint32_t), ctx.stream_a);
        carry_fixup_kernel_u160<<<num_segs, 1, 0, ctx.stream_a>>>(
            ctx.d_out, carry_in_lo, carry_in_mid, carry_in_hi,
            carry_out_lo, carry_out_mid, carry_out_hi,
            ctx.N, num_segs, ctx.d_carry_escape);
#else
        cudaMemsetAsync(carry_out, 0, num_segs * sizeof(int64_t), ctx.stream_a);
        carry_fixup_kernel<<<num_segs, 1, 0, ctx.stream_a>>>(
            ctx.d_out, carry_in, carry_out, ctx.N, num_segs, ctx.d_carry_escape);
#endif
        int escaped = 0;
        cudaMemcpyAsync(&escaped, ctx.d_carry_escape, sizeof(int),
                        cudaMemcpyDeviceToHost, ctx.stream_a);
        cudaStreamSynchronize(ctx.stream_a);
        if (!escaped)
            break;
#if defined(NATIVE_HOST_LIMBS)
        uint64_t* t_lo = carry_in_lo;  carry_in_lo = carry_out_lo;  carry_out_lo = t_lo;
        uint64_t* t_mid = carry_in_mid; carry_in_mid = carry_out_mid; carry_out_mid = t_mid;
        uint32_t* t_hi = carry_in_hi;  carry_in_hi = carry_out_hi;  carry_out_hi = t_hi;
#else
        int64_t* tmp = carry_in;
        carry_in = carry_out;
        carry_out = tmp;
#endif
    }

    if (prof.on)
        timing.carry_ms = prof.carry_timer->toc(ctx.stream_a);

    if (prof.on)
        prof.d2h_timer->tic(0);

    cudaMemcpy(C_out.data(), ctx.d_out,
            (ctx.N + 1) * sizeof(OutputLimbType), cudaMemcpyDeviceToHost);

    if (prof.on)
        timing.d2h_ms = prof.d2h_timer->toc(0);

    if (prof.on)
        cudaEventRecord(prof.execute_stop, 0);

    if (prof.on) {
        prof.fill(timing);
    }

    if (timing_out)
        *timing_out = timing;

#ifdef TIMING
    static bool header_written = false;

    ofstream file("ntt_timing.csv", ios::app);

    if (!header_written) {
        file << "N,L_A,L_B,INGRESS_FWD,H2D,FWD_PAD_NTT,FWD_A,FWD_B,MUL,INTT,CRT,CARRY,D2H,TOTAL\n";
        header_written = true;
    }

    file << ctx.N << ","
        << ctx.L_A << ","
        << ctx.L_B << ","
        << timing.ingress_fwd_ms << ","
        << timing.h2d_ms << ","
        << timing.fwd_pad_ntt_ms << ","
        << timing.fwd_pad_ntt_a_ms << ","
        << timing.fwd_pad_ntt_b_ms << ","
        << timing.pointwise_mul_ms << ","
        << timing.intt_ms << ","
        << timing.crt_ms << ","
        << timing.carry_ms << ","
        << timing.d2h_ms << ","
        << timing.total_ms << "\n";
#endif
}

void cleanup_ntt_context(NTTContext &ctx) {
    for (int i = 0; i < NUM_MODULI; i++) {
        cudaFree(ctx.a_dev[i]);
        cudaFree(ctx.b_dev[i]);
        cudaFree(ctx.c_dev[i]);
    }
    cudaStreamDestroy(ctx.stream_a);
    cudaStreamDestroy(ctx.stream_b);
    cudaFree(ctx.a_raw_dev);
    cudaFree(ctx.b_raw_dev);
#if defined(NATIVE_HOST_LIMBS)
    cudaFree(ctx.d_C_lo);
    cudaFree(ctx.d_C_mid);
    cudaFree(ctx.d_C_hi32);
#else
    cudaFree(ctx.d_C_hi);
    cudaFree(ctx.d_C_lo);
#endif
    cudaFree(ctx.d_out);
#if defined(NATIVE_HOST_LIMBS)
    cudaFree(ctx.d_seg_carry_lo);
    cudaFree(ctx.d_seg_carry_mid);
    cudaFree(ctx.d_seg_carry_hi);
    cudaFree(ctx.d_seg_carry_aux_lo);
    cudaFree(ctx.d_seg_carry_aux_mid);
    cudaFree(ctx.d_seg_carry_aux_hi);
#else
    cudaFree(ctx.d_seg_carry);
    cudaFree(ctx.d_seg_carry_aux);
#endif
    cudaFree(ctx.d_carry_escape);
}

void cleanup_ntt_precomputed(NTTPrecomputed &pre) {
    if (!pre.gpu_uploaded)
        return;

    for (int i = 0; i < NUM_MODULI; i++) {
        cudaFree(pre.forward_omega_dev[i]);
        cudaFree(pre.inverse_omega_dev[i]);
        cudaFree(pre.modulus_dev[i]);
        cudaFree(pre.ninv_dev[i]);
        pre.forward_omega_dev[i] = nullptr;
        pre.inverse_omega_dev[i] = nullptr;
        pre.modulus_dev[i] = nullptr;
        pre.ninv_dev[i] = nullptr;
    }
    pre.gpu_uploaded = false;
}