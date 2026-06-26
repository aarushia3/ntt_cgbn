// Carry propagation unit tests: 128-bit CRT coefficients -> 32-bit output limbs.
// Independent of LIMB_BITS / NUM_MODULI (same kernels in 32- and 64-bit builds).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "carry_prop.h"
#include "config.h"

using namespace std;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

// Serial CPU reference — matches scripts/carry.py.
static vector<OutputLimbType> cpu_carry_prop(
    const vector<uint64_t>& C_hi,
    const vector<uint64_t>& C_lo,
    size_t N)
{
    vector<OutputLimbType> out(N, 0);
    __int128 carry = 0;

    for (size_t i = 0; i < N; i++) {
        __int128 val = ((__uint128_t)C_hi[i] << 64) | C_lo[i];
        __int128 temp = val + carry;
        out[i] = (OutputLimbType)(temp & OUTPUT_LIMB_MASK);
        carry  = temp >> OUTPUT_LIMB_BITS;
    }
    return out;
}

static vector<OutputLimbType> run_carry_prop_gpu(
    const vector<uint64_t>& h_C_hi,
    const vector<uint64_t>& h_C_lo,
    size_t N)
{
    uint64_t *d_C_hi, *d_C_lo;
    cudaMalloc(&d_C_hi, N * sizeof(uint64_t));
    cudaMalloc(&d_C_lo, N * sizeof(uint64_t));
    cudaMemcpy(d_C_hi, h_C_hi.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_lo, h_C_lo.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    OutputLimbType* d_out;
    cudaMalloc(&d_out, N * sizeof(OutputLimbType));
    cudaMemset(d_out, 0, N * sizeof(OutputLimbType));

    size_t num_segs = (N + CARRY_SEG - 1) / CARRY_SEG;
    int64_t* d_seg_carry;
    int64_t* d_seg_carry_aux;
    cudaMalloc(&d_seg_carry, num_segs * sizeof(int64_t));
    cudaMalloc(&d_seg_carry_aux, num_segs * sizeof(int64_t));

    int64_t* carry_in = d_seg_carry;
    int64_t* carry_out = d_seg_carry_aux;

    carry_intra_segment_kernel<<<num_segs, 1>>>(
        d_C_hi, d_C_lo, d_out, carry_in, N);

    carry_inter_segment_kernel<<<1, 1>>>(carry_in, num_segs);

    int* d_escape;
    cudaMalloc(&d_escape, sizeof(int));
    for (;;) {
        cudaMemset(d_escape, 0, sizeof(int));
        cudaMemset(carry_out, 0, num_segs * sizeof(int64_t));
        carry_fixup_kernel<<<num_segs, 1>>>(
            d_out, carry_in, carry_out, N, num_segs, d_escape);
        int escaped = 0;
        cudaMemcpy(&escaped, d_escape, sizeof(int), cudaMemcpyDeviceToHost);
        if (!escaped)
            break;
        int64_t* tmp = carry_in;
        carry_in = carry_out;
        carry_out = tmp;
    }
    cudaFree(d_escape);

    cudaDeviceSynchronize();

    vector<OutputLimbType> out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(OutputLimbType), cudaMemcpyDeviceToHost);

    cudaFree(d_C_hi);
    cudaFree(d_C_lo);
    cudaFree(d_out);
    cudaFree(d_seg_carry);
    cudaFree(d_seg_carry_aux);
    return out;
}

static void to_hilo(unsigned __int128 val, uint64_t& hi, uint64_t& lo) {
    lo = (uint64_t)val;
    hi = (uint64_t)(val >> 64);
}

static bool check_gpu_cpu(
    const vector<uint64_t>& C_hi,
    const vector<uint64_t>& C_lo,
    size_t N,
    const char* name)
{
    auto gpu = run_carry_prop_gpu(C_hi, C_lo, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, N);
    bool ok = (gpu == cpu);
    if (!ok) {
        for (size_t i = 0; i < N && i < 8; i++) {
            if (gpu[i] != cpu[i])
                printf("  [%s] mismatch k=%zu gpu=%u cpu=%u\n",
                       name, i, gpu[i], cpu[i]);
        }
    }
    check(name, ok);
    return ok;
}

#if defined(NATIVE_HOST_LIMBS)

static void to_u160(unsigned __int128 val, uint64_t& lo, uint64_t& mid, uint32_t& hi) {
    lo = (uint64_t)val;
    mid = (uint64_t)(val >> 64);
    hi = 0;
}

static void to_u160_full(uint64_t lo_in, uint64_t mid_in, uint32_t hi_in,
                         uint64_t& lo, uint64_t& mid, uint32_t& hi) {
    lo = lo_in; mid = mid_in; hi = hi_in;
}

static vector<OutputLimbType> cpu_carry_prop_u160(
    const vector<uint64_t>& C_lo,
    const vector<uint64_t>& C_mid,
    const vector<uint32_t>& C_hi,
    size_t N)
{
    vector<OutputLimbType> out(N, 0);
    uint64_t c_lo = 0, c_mid = 0;
    uint32_t c_hi = 0;

    for (size_t i = 0; i < N; i++) {
        uint64_t lo = C_lo[i], mid = C_mid[i];
        uint32_t hi = C_hi[i];

        uint64_t old = lo;
        lo += c_lo;
        uint64_t carry = lo < old ? 1ULL : 0ULL;
        old = mid;
        mid += c_mid + carry;
        carry = (mid < old || (carry && mid == old)) ? 1ULL : 0ULL;
        hi += c_hi + (uint32_t)carry;

        out[i] = (OutputLimbType)lo;
        c_lo = mid;
        c_mid = (uint64_t)hi;
        c_hi = 0;
    }
    return out;
}

static vector<OutputLimbType> run_carry_prop_gpu_u160(
    const vector<uint64_t>& h_C_lo,
    const vector<uint64_t>& h_C_mid,
    const vector<uint32_t>& h_C_hi,
    size_t N)
{
    uint64_t *d_C_lo, *d_C_mid;
    uint32_t *d_C_hi;
    cudaMalloc(&d_C_lo, N * sizeof(uint64_t));
    cudaMalloc(&d_C_mid, N * sizeof(uint64_t));
    cudaMalloc(&d_C_hi, N * sizeof(uint32_t));
    cudaMemcpy(d_C_lo, h_C_lo.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_mid, h_C_mid.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_hi, h_C_hi.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    OutputLimbType* d_out;
    cudaMalloc(&d_out, N * sizeof(OutputLimbType));
    cudaMemset(d_out, 0, N * sizeof(OutputLimbType));

    size_t num_segs = (N + CARRY_SEG - 1) / CARRY_SEG;
    uint64_t *d_seg_carry_lo, *d_seg_carry_mid, *d_seg_carry_aux_lo, *d_seg_carry_aux_mid;
    uint32_t *d_seg_carry_hi, *d_seg_carry_aux_hi;
    cudaMalloc(&d_seg_carry_lo, num_segs * sizeof(uint64_t));
    cudaMalloc(&d_seg_carry_mid, num_segs * sizeof(uint64_t));
    cudaMalloc(&d_seg_carry_hi, num_segs * sizeof(uint32_t));
    cudaMalloc(&d_seg_carry_aux_lo, num_segs * sizeof(uint64_t));
    cudaMalloc(&d_seg_carry_aux_mid, num_segs * sizeof(uint64_t));
    cudaMalloc(&d_seg_carry_aux_hi, num_segs * sizeof(uint32_t));

    uint64_t* carry_in_lo  = d_seg_carry_lo;
    uint64_t* carry_in_mid = d_seg_carry_mid;
    uint32_t* carry_in_hi  = d_seg_carry_hi;
    uint64_t* carry_out_lo  = d_seg_carry_aux_lo;
    uint64_t* carry_out_mid = d_seg_carry_aux_mid;
    uint32_t* carry_out_hi  = d_seg_carry_aux_hi;

    carry_intra_segment_kernel_u160<<<num_segs, 1>>>(
        d_C_lo, d_C_mid, d_C_hi, d_out,
        carry_in_lo, carry_in_mid, carry_in_hi, N);
    carry_inter_segment_kernel_u160<<<1, 1>>>(
        carry_in_lo, carry_in_mid, carry_in_hi, num_segs);

    int* d_escape;
    cudaMalloc(&d_escape, sizeof(int));
    for (;;) {
        cudaMemset(d_escape, 0, sizeof(int));
        cudaMemset(carry_out_lo, 0, num_segs * sizeof(uint64_t));
        cudaMemset(carry_out_mid, 0, num_segs * sizeof(uint64_t));
        cudaMemset(carry_out_hi, 0, num_segs * sizeof(uint32_t));
        carry_fixup_kernel_u160<<<num_segs, 1>>>(
            d_out, carry_in_lo, carry_in_mid, carry_in_hi,
            carry_out_lo, carry_out_mid, carry_out_hi,
            N, num_segs, d_escape);
        int escaped = 0;
        cudaMemcpy(&escaped, d_escape, sizeof(int), cudaMemcpyDeviceToHost);
        if (!escaped)
            break;
        uint64_t* t_lo = carry_in_lo;  carry_in_lo = carry_out_lo;  carry_out_lo = t_lo;
        uint64_t* t_mid = carry_in_mid; carry_in_mid = carry_out_mid; carry_out_mid = t_mid;
        uint32_t* t_hi = carry_in_hi;  carry_in_hi = carry_out_hi;  carry_out_hi = t_hi;
    }
    cudaDeviceSynchronize();

    vector<OutputLimbType> out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(OutputLimbType), cudaMemcpyDeviceToHost);

    cudaFree(d_C_lo); cudaFree(d_C_mid); cudaFree(d_C_hi);
    cudaFree(d_out);
    cudaFree(d_seg_carry_lo); cudaFree(d_seg_carry_mid); cudaFree(d_seg_carry_hi);
    cudaFree(d_seg_carry_aux_lo); cudaFree(d_seg_carry_aux_mid); cudaFree(d_seg_carry_aux_hi);
    cudaFree(d_escape);
    return out;
}

static bool check_gpu_cpu_u160(
    const vector<uint64_t>& C_lo,
    const vector<uint64_t>& C_mid,
    const vector<uint32_t>& C_hi,
    size_t N,
    const char* name)
{
    auto gpu = run_carry_prop_gpu_u160(C_lo, C_mid, C_hi, N);
    auto cpu = cpu_carry_prop_u160(C_lo, C_mid, C_hi, N);
    bool ok = (gpu == cpu);
    if (!ok) {
        for (size_t i = 0; i < N && i < 8; i++) {
            if (gpu[i] != cpu[i])
                printf("  [%s] mismatch k=%zu gpu=%llu cpu=%llu\n",
                       name, i, (unsigned long long)gpu[i], (unsigned long long)cpu[i]);
        }
    }
    check(name, ok);
    return ok;
}

static void test_u160_all_zeros() {
    size_t N = 64;
    vector<uint64_t> C_lo(N, 0), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 all zeros");
}

static void test_u160_small_values_no_carry() {
    size_t N = 64;
    vector<uint64_t> C_lo(N), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    for (size_t i = 0; i < N; i++) C_lo[i] = i + 1;
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 small values no carry");
}

static void test_u160_single_overflow() {
    size_t N = 64;
    vector<uint64_t> C_lo(N, 0), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    C_mid[0] = 1;
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 single overflow");
}

static void test_u160_max_limb_values() {
    size_t N = 1 << 10;
    vector<uint64_t> C_lo(N, UINT64_MAX), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 max limb values");
}

static void test_u160_wide_160bit_coeff() {
    size_t N = 16;
    vector<uint64_t> C_lo(N, 0), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    C_lo[0] = 12345;
    C_hi[0] = (uint32_t)(1U << 31);
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 wide 160-bit coefficient");
}

static void test_u160_160bit_max() {
    size_t N = 8;
    vector<uint64_t> C_lo(N, UINT64_MAX), C_mid(N, UINT64_MAX);
    vector<uint32_t> C_hi(N, UINT32_MAX);
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 max (2^160-1)");
}

static void test_u160_128bit_boundary() {
    size_t N = 16;
    vector<uint64_t> C_lo(N, 1), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    C_lo[0] = 1;
    C_mid[0] = 1;
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 128-bit boundary");
}

static void test_u160_cross_segment_carry() {
    size_t N = CARRY_SEG * 2;
    vector<uint64_t> C_lo(N, 0), C_mid(N, 0);
    vector<uint32_t> C_hi(N, 0);
    C_lo[CARRY_SEG - 1] = UINT64_MAX;
    C_mid[CARRY_SEG - 1] = 1;
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 cross-segment carry");
}

static void test_u160_multi_segment_random() {
    size_t N = CARRY_SEG * 8;
    vector<uint64_t> C_lo(N), C_mid(N);
    vector<uint32_t> C_hi(N);
    for (size_t i = 0; i < N; i++) {
        unsigned __int128 val;
        switch (i % 5) {
        case 0: val = (unsigned __int128)(i * 12345 + 1); break;
        case 1: val = ((unsigned __int128)1 << 120) + (i % 1000); break;
        case 2: val = UINT64_MAX; break;
        case 3: val = ((unsigned __int128)(i % 17) << 64) + (i * 99991); break;
        default: val = ((unsigned __int128)1 << 155) + i; break;
        }
        to_u160(val, C_lo[i], C_mid[i], C_hi[i]);
    }
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 multi-segment random (N=8k)");
}

static void test_u160_large_random() {
    size_t N = 1 << 12;
    vector<uint64_t> C_lo(N), C_mid(N);
    vector<uint32_t> C_hi(N);
    for (size_t i = 0; i < N; i++) {
        unsigned __int128 val;
        switch (i % 4) {
        case 0: val = (unsigned __int128)(i * 12345 + 1); break;
        case 1: val = ((unsigned __int128)1 << 100) + (i % 1000); break;
        case 2: val = UINT64_MAX; break;
        default: val = ((unsigned __int128)(i % 17) << 64) + (i * 99991); break;
        }
        to_u160(val, C_lo[i], C_mid[i], C_hi[i]);
    }
    check_gpu_cpu_u160(C_lo, C_mid, C_hi, N, "u160 large random (N=2^12)");
}

#endif // NATIVE_HOST_LIMBS

static void test_all_zeros() {
    size_t N = 64;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    check_gpu_cpu(C_hi, C_lo, N, "all zeros");
}

static void test_small_values_no_carry() {
    size_t N = 64;
    vector<uint64_t> C_hi(N, 0), C_lo(N);
    for (size_t i = 0; i < N; i++) C_lo[i] = (uint64_t)(i + 1);
    check_gpu_cpu(C_hi, C_lo, N, "small values no carry");
}

static void test_single_overflow() {
    size_t N = 64;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    C_lo[0] = OUTPUT_LIMB_MASK + 1;
    check_gpu_cpu(C_hi, C_lo, N, "single overflow into next limb");
}

static void test_max_limb_values() {
    size_t N = 1 << 10;
    vector<uint64_t> C_hi(N, 0), C_lo(N, OUTPUT_LIMB_MASK);
    check_gpu_cpu(C_hi, C_lo, N, "max limb values (heavy carry chain)");
}

static void test_wide_128bit_coeff() {
    // Coefficient uses high word — carry must see full 128-bit value.
    size_t N = 16;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    unsigned __int128 val = (unsigned __int128)1 << 70;
    val += 12345;
    to_hilo(val, C_hi[0], C_lo[0]);
    check_gpu_cpu(C_hi, C_lo, N, "wide 128-bit coefficient (hi != 0)");
}

static void test_cross_segment_carry() {
    size_t N = CARRY_SEG * 2;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    C_lo[CARRY_SEG - 1] = OUTPUT_LIMB_MASK + 1;
    check_gpu_cpu(C_hi, C_lo, N, "cross-segment carry");
}

static void test_large_random() {
    size_t N = 1 << 16;
    vector<uint64_t> C_hi(N), C_lo(N);
    for (size_t i = 0; i < N; i++) {
        unsigned __int128 val;
        if (i % 4 == 0)
            val = (unsigned __int128)(i * 12345 + 1);
        else if (i % 4 == 1)
            val = ((unsigned __int128)1 << 70) + (i % 1000);
        else if (i % 4 == 2)
            val = OUTPUT_LIMB_MASK;
        else
            val = ((unsigned __int128)(i % 17) << 64) + (i * 99991);
        to_hilo(val, C_hi[i], C_lo[i]);
    }
    check_gpu_cpu(C_hi, C_lo, N, "large random (N=2^16)");
}

int main() {
    printf("=== carry_prop unit tests (LIMB_BITS=%d, output=%d-bit limbs) ===\n",
           LIMB_BITS, OUTPUT_LIMB_BITS);

#if defined(NATIVE_HOST_LIMBS)
    static_assert(OUTPUT_LIMB_BITS == 64, "carry output must be 64-bit limbs");
    static_assert(CRT_COEFF_BITS == 160, "native CRT must be 160-bit");

    test_u160_all_zeros();
    test_u160_small_values_no_carry();
    test_u160_single_overflow();
    test_u160_max_limb_values();
    test_u160_wide_160bit_coeff();
    test_u160_160bit_max();
    test_u160_128bit_boundary();
    test_u160_cross_segment_carry();
    test_u160_multi_segment_random();
    test_u160_large_random();
#else
    static_assert(OUTPUT_LIMB_BITS == 32, "carry output must be 32-bit limbs");
    static_assert(sizeof(OutputLimbType) == 4, "OutputLimbType must be uint32_t");

    test_all_zeros();
    test_small_values_no_carry();
    test_single_overflow();
    test_max_limb_values();
    test_wide_128bit_coeff();
    test_cross_segment_carry();
    test_large_random();
#endif

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
