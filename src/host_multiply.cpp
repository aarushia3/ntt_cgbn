// multiplication pipeline: split -> NTs modulo several primes -> pointwise mul
// -> CRT reconstruction -> carry propagation

// need to choose a limb / base b = 2^w, probably w = 32 to use CGBN
// A, B -> need to generate arrays of L limbs in base b. Convolution length <=
// L_A + L_B - 1 choose N - next power of 2 >= L_A + L_B - 1. Use GPU-NTT to
// compute transform of length N. We need several primes here. Single NTT module
// one prime p gives residues module p. We need to pick enough moduli p_i s. t.
// product p_i > (b-1)^2 * L. reconstruct each coefficient. CGBN can be used
// here. after crt reconstruction, do base b carry propagation.

#include "../include/gpu_cgbn.h"
#include "../include/gpu_ntt.h"
#include "../include/multiply.h"
#include "config.h"

#include <algorithm>

using namespace std;

// parameters
// constexpr unsigned LIMB_BITS = 32;
using limb_t = uint32_t; // each limb is stored in 32-bit containers
// constexpr uint32_t BASE = (1ULL << LIMB_BITS); // base b = 2^w

// host functions
void host_multiply_merge(const vector<limb_t> &A, const vector<limb_t> &B, vector<limb_t> &C) {
    size_t L_A = A.size();
    size_t L_B = B.size();
    size_t L_C = L_A + L_B - 1;  // not sure what the length of outputs will be? paper suggests L_A = L_B = L_C

    // pad to NTT length, must be power of 2
    size_t N = 1;
    while (N < L_C)
        N <<= 1;

    vector<uint32_t> A_pad(L_A, 0), B_pad(L_B, 0);
    copy(A.begin(), A.end(), A_pad.begin());
    copy(B.begin(), B.end(), B_pad.begin());

    // print A_pad and B_pad
    cout << "[Host] Padded A: ";
    for (auto x : A_pad)
        cout << x << " ";
    cout << endl;

    cout << "[Host] Padded B: ";
    for (auto x : B_pad)
        cout << x << " ";
    cout << endl;

    // ntt_merge_forward should return 4 versions of the NTT. don't do the for loop
    vector<vector<uint32_t>> A_mod(NUM_MODULI, vector<uint32_t>(N));
    vector<vector<uint32_t>> B_mod(NUM_MODULI, vector<uint32_t>(N));
    ntt_merge_forward(A_pad, A_mod);
    ntt_merge_forward(B_pad, B_mod);

    // pointwise multiplication for each of the elements of A_mod, B_mod
    vector<vector<uint32_t>> C_mod;
    gpu_pointwise_multiply(A_mod, B_mod, C_mod);

    // gpu_ntt_inverse calls, should do the inverse ntt computation for all 4 C_mod vectors
    vector<vector<uint32_t>> C_recovered;
    gpu_ntt_inverse(C_mod, C_recovered);

    // CRT recombination (CGBN)
    // vector<__uint128_t> C_big(N);
    // gpu_crt_reconstruct(C_mod, C_big, MODULI, NUM_MODULI);

    // Carry propagation (CGBN)
    // C.resize(L_C + 1);
    // gpu_carry_propagate(C_big, C, BASE);

    // trim
    while (C.size() > 1 && C.back() == 0)
      C.pop_back();
}
