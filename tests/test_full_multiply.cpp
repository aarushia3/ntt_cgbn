#include <vector>
#include <iostream>
#include <random>
#include <cassert>
#include <limits>
#include <chrono>
#include <gmp.h>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "../include/multiply.h"
#include "../include/config.h"

using namespace std;

// ---------------- ANSI COLORS ----------------
#define GREEN_BOLD "\033[1;32m"
#define RED_BOLD   "\033[1;31m"
#define YELLOW     "\033[33m"
#define RESET      "\033[0m"

#if !defined(NATIVE_HOST_LIMBS)

// ---------------- CPU REFERENCE MULTIPLY ----------------
// Bigint schoolbook multiply with full carry
vector<OutputLimbType> cpu_schoolbook_mul(
    const vector<uint32_t>& A,
    const vector<uint32_t>& B)
{
    size_t n = A.size(), m = B.size();
    vector<unsigned __int128> tmp(n + m, 0);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            tmp[i + j] += (unsigned __int128)A[i] * B[j];
        }
    }

    vector<OutputLimbType> C(tmp.size());
    unsigned __int128 carry = 0;
    const unsigned SHIFT = OUTPUT_LIMB_BITS;

    for (size_t i = 0; i < tmp.size(); i++) {
        tmp[i] += carry;
        C[i] = (OutputLimbType)(tmp[i] & (((unsigned __int128)1 << SHIFT) - 1));
        carry = tmp[i] >> SHIFT;
    }

    while (C.size() > 1 && C.back() == 0)
        C.pop_back();

    size_t linear_len = A.size() + B.size(); // 2*L-1
    if (C.size() < linear_len)
        C.resize(linear_len);

    return C;
}

// ---------------- RANDOM INPUT GENERATOR ----------------
vector<uint32_t> random_limbs(size_t n, uint64_t seed)
{
    mt19937_64 rng(seed);
    vector<uint32_t> v(n);

    for (size_t i = 0; i < n; i++) {
        if (i % 8 == 0) v[i] = 0; // edge case
        else if (i % 8 == 1) v[i] = 1;
        // else if (i % 8 == 2) v[i] = numeric_limits<TestDataTypeUint>::max();
        else v[i] = (uint32_t)rng() % (1ULL << 30);
    }
    return v;
}

// ---------------- COMPARE & REPORT ----------------
bool compare_vectors(const vector<OutputLimbType>& A,
                     const vector<OutputLimbType>& B)
{
    const size_t MAX_REPORT = 4;

    if (A.size() != B.size()) {
        cout << RED_BOLD << "[FAIL] Size mismatch: "
             << A.size() << " vs " << B.size() << RESET << "\n";
        return false;
    }

    bool ok = true;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < A.size(); i++) {
        if (A[i] != B[i]) {
            if (mismatch_count < MAX_REPORT) {
                cout << RED_BOLD << "[MISMATCH] index " << i
                     << " GPU=" << A[i]
                     << " CPU=" << B[i] << RESET << "\n";
            }
            mismatch_count++;
            ok = false;

            if (mismatch_count == MAX_REPORT) {
                cout << RED_BOLD
                     << "... further mismatches not reported ..."
                     << RESET << "\n";
            }
        }
    }

    if (mismatch_count > 0) {
        cout << RED_BOLD << "[SUMMARY] Total mismatches: "
             << mismatch_count << RESET << "\n";
    }

    return ok;
}

// ---------------- GMP HELPERS ----------------
void to_mpz(mpz_t out, const vector<OutputLimbType>& v)
{
    mpz_set_ui(out, 0);
    const size_t limb_bits = OUTPUT_LIMB_BITS;

    for (ssize_t i = (ssize_t)v.size() - 1; i >= 0; i--) {
        mpz_mul_2exp(out, out, limb_bits);
        mpz_add_ui(out, out, v[i]);
    }
}

vector<OutputLimbType> from_mpz(const mpz_t x, size_t expected_limbs)
{
    vector<OutputLimbType> out(expected_limbs, 0);

    mpz_t tmp;
    mpz_init_set(tmp, x);

    const unsigned limb_bits = OUTPUT_LIMB_BITS;

    for (size_t i = 0; i < expected_limbs; i++) {
        out[i] = (OutputLimbType) mpz_get_ui(tmp);
        mpz_fdiv_q_2exp(tmp, tmp, limb_bits);
    }

    mpz_clear(tmp);
    return out;
}

void limbs_to_mpz(mpz_t result,
                  const uint32_t* limbs,
                  size_t n)
{
    mpz_import(result, n, -1, sizeof(OutputLimbType), 0, 0, limbs);
}

vector<OutputLimbType> fast_from_mpz(mpz_t value, size_t expected_limbs)
{
    vector<OutputLimbType> out(expected_limbs);

    size_t count = 0;
    mpz_export(
        out.data(),
        &count,
        -1,
        sizeof(OutputLimbType),
        0,
        0,
        value
    );

    out.resize(expected_limbs, 0);

    return out;
}

vector<OutputLimbType> gmp_mul(
    const vector<uint32_t>& A,
    const vector<uint32_t>& B)
{
    using clock = std::chrono::high_resolution_clock;

    auto t_total_start = clock::now();

    mpz_t a, b, c;
    mpz_init(a);
    mpz_init(b);
    mpz_init(c);

    // ---- Import A ----
    auto t1 = clock::now();
    mpz_import(a, A.size(), -1, sizeof(uint32_t), 0, 0, A.data());
    auto t2 = clock::now();

    // ---- Import B ----
    auto t3 = clock::now();
    mpz_import(b, B.size(), -1, sizeof(uint32_t), 0, 0, B.data());
    auto t4 = clock::now();

    // ---- Multiply ----
    auto t5 = clock::now();
    mpz_mul(c, a, b);
    auto t6 = clock::now();

    // ---- Export ----
    auto t7 = clock::now();
    auto out = fast_from_mpz(c, A.size() + B.size());
    auto t8 = clock::now();

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);

    auto t_total_end = clock::now();

    // ---- Compute durations ----
    double importA_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double importB_us = std::chrono::duration<double, std::micro>(t4 - t3).count();
    double multiply_us = std::chrono::duration<double, std::micro>(t6 - t5).count();
    double export_us = std::chrono::duration<double, std::micro>(t8 - t7).count();
    double total_us = std::chrono::duration<double, std::micro>(t_total_end - t_total_start).count();

    // ---- Append to CSV ----
    static bool wrote_header = false;
    std::ofstream csv("gmp_profile.csv", std::ios::app);

    if (!wrote_header) {
        csv << "sizeA,sizeB,importA_us,importB_us,multiply_us,export_us,total_us\n";
        wrote_header = true;
    }

    csv << A.size() << ","
        << B.size() << ","
        << importA_us << ","
        << importB_us << ","
        << multiply_us << ","
        << export_us << ","
        << total_us << "\n";

    return out;
}

// ---------------- PIPELINE TEST ----------------
void test_full_pipeline(size_t L)
{
    cout << YELLOW << "\n[Test] Full multiply pipeline, L = "
         << L << " limbs" << RESET << "\n";

    vector<uint32_t> A = random_limbs(L, 1234);
    vector<uint32_t> B = random_limbs(L, 5678);

    // vector<OutputLimbType> A = {1,2,3,4,5,6,7,8};
    // vector<OutputLimbType> B = {9,10,11,12,13,14,15,16};

    // vector<uint32_t> A = {1,2,3,4};
    // vector<uint32_t> B = {5,6,7,8};

    // GPU pipeline
    vector<OutputLimbType> C_gpu;
    chrono::duration<double, milli> duration;
    host_multiply_merge(A, B, C_gpu, duration);

    // CPU reference
    vector<OutputLimbType> C_cpu = cpu_schoolbook_mul(A, B);

    // Compare
    bool ok = compare_vectors(C_gpu, C_cpu);

    if (ok)
        cout << GREEN_BOLD << "[PASS] Full pipeline correct\n" << RESET;
    else
        cout << RED_BOLD << "[FAIL] Pipeline incorrect\n" << RESET;
}

void test_simple() {
    chrono::duration<double, milli> dur;
    bool all_ok = true;
    auto check = [&](const vector<uint32_t>& A,
                     const vector<uint32_t>& B,
                     const string& label) {
        vector<OutputLimbType> C_gpu, C_ref;
        host_multiply_merge(A, B, C_gpu, dur);
        C_ref = cpu_schoolbook_mul(A, B);
        bool ok = compare_vectors(C_gpu, C_ref);
        cout << (ok ? GREEN_BOLD "[PASS] " : RED_BOLD "[FAIL] ") << label << RESET << "\n";
        all_ok &= ok;
    };

    check({1, 0, 0, 0}, {1, 0, 0, 0}, "1 x 1");
    check({0, 0, 0, 0}, {42, 0, 0, 0}, "0 x 42");
}

void test_identities(size_t L) {
    cout << YELLOW << "\n[Test] Identities, L = " << L << " limbs" << RESET << "\n";
    vector<uint32_t> Z(L, 0);
    vector<uint32_t> O(L, 1);
    vector<OutputLimbType> R;

    chrono::duration<double, milli> duration;
    host_multiply_merge(Z, Z, R, duration);
    assert(all_of(R.begin(), R.end(), [](auto x){return x==0;}));

    host_multiply_merge(O, Z, R, duration);
    assert(all_of(R.begin(), R.end(), [](auto x){return x==0;}));

    host_multiply_merge(O, O, R, duration);
    assert(R[0] == 1);
    cout << GREEN_BOLD << "[PASS] Identities correct\n" << RESET;
}

void test_root_of_unity(size_t L)
{
    cout << YELLOW
         << "\n[Test] Root-of-unity probe, L = "
         << L
         << RESET << "\n";

    vector<uint32_t> A(L, 0);
    vector<uint32_t> B(L, 0);

    // probe vector
    A[1] = 1;

    // neutral multiplier so A survives pipeline
    B[0] = 1;

    vector<OutputLimbType> C;
    chrono::duration<double, milli> duration;

    cout << "[INFO] Expect forward NTT of A to resemble:\n"
     << "       [1, w, w^2, ...] up to output permutation\n"
     << "       check debug logs for:\n"
     << "       - first entry == 1\n"
     << "       - -1 mod p appears somewhere\n"
     << "       - entries appear distinct\n";

    host_multiply_merge(A, B, C, duration);

    // convolution identity sanity:
    // multiplying by [1,0,0,...] should reproduce A
    vector<OutputLimbType> expected(A.size() + B.size(), 0);
    expected[1] = 1;

    bool ok = compare_vectors(C, expected);

    if (ok)
        cout << GREEN_BOLD
             << "[PASS] Root probe completed "
             << "(inspect debug NTT output)"
             << RESET << "\n";
    else
        cout << RED_BOLD
             << "[FAIL] Root probe convolution incorrect"
             << RESET << "\n";
}

// ---------------- BENCHMARK ----------------
void benchmark_vs_gmp(size_t L)
{
    cout << YELLOW << "\n[Benchmark] L = " << L << RESET << "\n";

    vector<uint32_t> A = random_limbs(L, 1234);
    vector<uint32_t> B = random_limbs(L, 5678);

    vector<OutputLimbType> C_gpu, C_gmp;

    // Warm-up
    vector<OutputLimbType> warm;
    chrono::duration<double, milli> duration;
    host_multiply_merge(A, B, warm, duration);
    C_gmp = gmp_mul(A, B);

    const int ITERS = 20;
    double gpu_time = 0.0, gmp_time = 0.0;
    double gpu_time_sq = 0.0, gmp_time_sq = 0.0;

    for (int i = 0; i < ITERS; i++) {
        chrono::duration<double, milli> duration;
        host_multiply_merge(A, B, C_gpu, duration);
        double t = duration.count();
        gpu_time += t;
        gpu_time_sq += t * t;
    }

    for (int i = 0; i < ITERS; i++) {
        auto t3 = chrono::high_resolution_clock::now();
        C_gmp = gmp_mul(A, B);
        auto t4 = chrono::high_resolution_clock::now();
        double t = chrono::duration<double, milli>(t4 - t3).count();
        gmp_time += t;
        gmp_time_sq += t * t;
    }

    double gpu_mean = gpu_time / ITERS;
    double gmp_mean = gmp_time / ITERS;

    double gpu_var = (gpu_time_sq / ITERS) - (gpu_mean * gpu_mean);
    double gmp_var = (gmp_time_sq / ITERS) - (gmp_mean * gmp_mean);

    double gpu_std = sqrt(gpu_var);
    double gmp_std = sqrt(gmp_var);

    cout << fixed << setprecision(3);

    cout << "GPU avg time: " << gpu_mean
        << " ms (stddev: " << gpu_std << ")\n";

    cout << "GMP avg time: " << gmp_mean
        << " ms (stddev: " << gmp_std << ")\n";

    bool ok = compare_vectors(C_gpu, C_gmp);
    cout << (ok ? GREEN_BOLD "[MATCH]\n" RESET : RED_BOLD "[MISMATCH]\n" RESET);
}

void profile_run(size_t L)
{
    vector<uint32_t> A = random_limbs(L, 1234);
    vector<uint32_t> B = random_limbs(L, 5678);
    vector<OutputLimbType> C;

    chrono::duration<double, milli> duration;

    // warmup
    host_multiply_merge(A, B, C, duration);

    // profile this run
    host_multiply_merge(A, B, C, duration);
}

#endif // !NATIVE_HOST_LIMBS

// ---------------- MAIN ----------------
#if defined(NATIVE_HOST_LIMBS)

vector<uint64_t> random_limbs_u64(size_t n, uint64_t seed) {
    mt19937_64 rng(seed);
    vector<uint64_t> v(n);
    for (size_t i = 0; i < n; i++) {
        if (i % 8 == 0) v[i] = 0;
        else if (i % 8 == 1) v[i] = 1;
        else if (i % 8 == 2) v[i] = UINT64_MAX;
        else v[i] = rng();
    }
    return v;
}

vector<uint64_t> gmp_mul_u64(const vector<uint64_t>& A, const vector<uint64_t>& B) {
    mpz_t a, b, c;
    mpz_init(a); mpz_init(b); mpz_init(c);
    mpz_import(a, A.size(), -1, sizeof(uint64_t), 0, 0, A.data());
    mpz_import(b, B.size(), -1, sizeof(uint64_t), 0, 0, B.data());
    mpz_mul(c, a, b);
    vector<uint64_t> out(A.size() + B.size());
    size_t count = 0;
    mpz_export(out.data(), &count, -1, sizeof(uint64_t), 0, 0, c);
    out.resize(A.size() + B.size(), 0);
    mpz_clear(a); mpz_clear(b); mpz_clear(c);
    return out;
}

bool compare_u64(const vector<uint64_t>& A, const vector<uint64_t>& B) {
    const size_t MAX_REPORT = 4;
    size_t n = max(A.size(), B.size());
    bool ok = true;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < n; i++) {
        uint64_t a = i < A.size() ? A[i] : 0;
        uint64_t b = i < B.size() ? B[i] : 0;
        if (a != b) {
            if (mismatch_count < MAX_REPORT) {
                cout << RED_BOLD << "  [MISMATCH] limb " << i
                     << " gpu=" << a << " gmp=" << b << RESET << "\n";
            }
            mismatch_count++;
            ok = false;
            if (mismatch_count == MAX_REPORT) {
                cout << RED_BOLD
                     << "  ... further mismatches not reported ..."
                     << RESET << "\n";
            }
        }
    }

    if (mismatch_count > 0) {
        cout << RED_BOLD << "  total mismatches: " << mismatch_count
             << RESET << "\n";
    }
    return ok;
}

void test_native_pipeline(size_t L) {
    cout << YELLOW << "\n[Test] 64-bit multiply L=" << L << RESET << "\n";
    auto A = random_limbs_u64(L, 1234);
    auto B = random_limbs_u64(L, 5678);
    vector<uint64_t> C_gpu, C_gmp;
    chrono::duration<double, milli> dur;
    host_multiply_merge_native(A, B, C_gpu, dur);
    cout << "  gpu time: " << fixed << setprecision(1) << dur.count() << " ms\n";
    auto t0 = chrono::high_resolution_clock::now();
    C_gmp = gmp_mul_u64(A, B);
    auto t1 = chrono::high_resolution_clock::now();
    double gmp_ms = chrono::duration<double, milli>(t1 - t0).count();
    cout << "  gmp time: " << fixed << setprecision(1) << gmp_ms << " ms\n";
    bool ok = compare_u64(C_gpu, C_gmp);
    cout << (ok ? GREEN_BOLD "[PASS]\n" : RED_BOLD "[FAIL]\n") << RESET;
    if (!ok) exit(1);
}

int main() {
    cout << YELLOW << "==== 64-BIT FULL MULTIPLY TEST ====\n" << RESET;
    test_native_pipeline(4);
    test_native_pipeline(16);
    test_native_pipeline(64);
    test_native_pipeline(256);
    test_native_pipeline(1 << 12);   // 4096 — cross-segment carry regression
    test_native_pipeline(1 << 10);   // 1024
    test_native_pipeline(1 << 16);   // 65536
    test_native_pipeline(1 << 20);   // 1048576
    cout << YELLOW << "==== TEST COMPLETE ====\n" << RESET;
    return 0;
}

#else

int main()
{
    #ifdef PROFILE
    
        cout << YELLOW << "==== PROFILE RUN ====\n" << RESET;
        profile_run(1ULL << 22);
        cout << YELLOW << "==== PROFILE COMPLETE ====\n" << RESET;
        return 0;
    
    #else

        cout << YELLOW << "==== FULL MULTIPLICATION PIPELINE TEST ====\n" << RESET;

        test_simple();

        test_identities(1ULL << 20);

        test_root_of_unity(8);

        test_full_pipeline(4);
        test_full_pipeline(8);
        test_full_pipeline(16);
        test_full_pipeline(64);

        test_full_pipeline(128);
        test_full_pipeline(2048);
        test_full_pipeline(10000);
        test_full_pipeline(1ULL << 15);

        benchmark_vs_gmp(4);
        benchmark_vs_gmp(256);
        benchmark_vs_gmp(1ULL << 12);
        benchmark_vs_gmp(1ULL << 15);
        benchmark_vs_gmp(1ULL << 20);
        benchmark_vs_gmp(1ULL << 22);

        cout << YELLOW << "\n==== TEST COMPLETE ====\n" << RESET;
        return 0;
        
    #endif
}

#endif // NATIVE_HOST_LIMBS
