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

#include <algorithm>

// parameters
constexpr unsigned LIMB_BITS = 32;
using limb_t = uint32_t; // each limb is stored in 32-bit containers
constexpr uint64_t BASE = (1ULL << LIMB_BITS); // base b = 2^w

// host functions
void host_multiply(const std::vector<limb_t> &A, const std::vector<limb_t> &B,
                   std::vector<limb_t> &C) {
  size_t L_A = A.size();
  size_t L_B = B.size();
  size_t L_C = L_A + L_B - 1;

  // constants
  constexpr int NUM_MODULI = 4;
  constexpr uint32_t MODULI[NUM_MODULI] = {2013265921, 1811939329, 2113929217,
                                           2013265921};

  // pad to NTT length
  size_t N = 1;
  while (N < L_C)
    N <<= 1;

  std::vector<uint32_t> A_pad(N, 0), B_pad(N, 0);
  std::copy(A.begin(), A.end(), A_pad.begin());
  std::copy(B.begin(), B.end(), B_pad.begin());

  // print A_pad and B_pad
  std::cout << "[Host] Padded A: ";
  for (auto x : A_pad)
    std::cout << x << " ";
  std::cout << std::endl;

  std::cout << "[Host] Padded B: ";
  for (auto x : B_pad)
    std::cout << x << " ";
  std::cout << std::endl;

  // run NTT for each modulus
  std::vector<std::vector<uint32_t>> C_mod(NUM_MODULI,
                                           std::vector<uint32_t>(N));
  for (int i = 0; i < NUM_MODULI; i++) {
    uint32_t p = MODULI[i];
    gpu_ntt_forward(A_pad, p);
    // print out A_pad after NTT
    std::cout << "[Host] NTT(A) mod " << p << ": ";
    for (auto x : A_pad)
      std::cout << x << " ";
    std::cout << std::endl;
    gpu_ntt_forward(B_pad, p);
    // print out B_pad after NTT
    std::cout << "[Host] NTT(B) mod " << p << ": ";
    for (auto x : B_pad)
      std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "================================" << std::endl;
    gpu_pointwise_multiply(A_pad, B_pad, C_mod[i], p);
    // gpu_ntt_inverse(C_mod[i], p);
  }

  // CRT recombination (CGBN)
  // std::vector<__uint128_t> C_big(N);
  // gpu_crt_reconstruct(C_mod, C_big, MODULI, NUM_MODULI);

  // Carry propagation (CGBN)
  // C.resize(L_C + 1);
  // gpu_carry_propagate(C_big, C, BASE);

  // trim
  while (C.size() > 1 && C.back() == 0)
    C.pop_back();
}
