#pragma once

#include "ntt.cuh"
#include "config.h"
#include "crt_gpu.h"
#include "carry_prop.h"
#include <vector>

using namespace std;
using namespace gpuntt;

#if LIMB_BITS == 64
    typedef Data64 TestDataType;
#else
    typedef Data32 TestDataType;
#endif

struct NTTContext {
    size_t N;
    int logN;

    InputLimbType* a_raw_dev = nullptr;
    InputLimbType* b_raw_dev = nullptr;
    size_t L_A = 0, L_B = 0;

    vector<TestDataType*> a_dev;
    vector<TestDataType*> b_dev;
    vector<TestDataType*> c_dev;

    vector<Root<TestDataType>*> forward_omega_dev;
    vector<Root<TestDataType>*> inverse_omega_dev;

    vector<Modulus<TestDataType>*> modulus_dev;
    vector<Ninverse<TestDataType>*> ninv_dev;

    cudaStream_t stream_a, stream_b;

#if defined(NATIVE_HOST_LIMBS)
    uint64_t* d_C_lo = nullptr;
    uint64_t* d_C_mid = nullptr;
    uint32_t* d_C_hi32 = nullptr;
#else
    uint64_t* d_C_hi = nullptr;
    uint64_t* d_C_lo = nullptr;
#endif

    OutputLimbType* d_out = nullptr;
#if defined(NATIVE_HOST_LIMBS)
    uint64_t* d_seg_carry_lo = nullptr;
    uint64_t* d_seg_carry_mid = nullptr;
    uint32_t* d_seg_carry_hi = nullptr;
    uint64_t* d_seg_carry_aux_lo = nullptr;
    uint64_t* d_seg_carry_aux_mid = nullptr;
    uint32_t* d_seg_carry_aux_hi = nullptr;
#else
    int64_t*    d_seg_carry = nullptr;
    int64_t*    d_seg_carry_aux = nullptr;
#endif
    int*        d_carry_escape = nullptr;
};

struct NTTPrecomputed {
	size_t N;
	int logN;
	vector<NTTParameters<TestDataType>> params;
	vector<vector<Root<TestDataType>>> forward_omega_host;
	vector<vector<Root<TestDataType>>> inverse_omega_host;
	CRTGarnerParams garner;
	bool gpu_uploaded = false;
	vector<Root<TestDataType>*>         forward_omega_dev;
	vector<Root<TestDataType>*>         inverse_omega_dev;
	vector<Modulus<TestDataType>*>    modulus_dev;
	vector<Ninverse<TestDataType>*> ninv_dev;
};

struct PrecomputeTiming {
	float factors_ms = 0.0f;
	float params_ms = 0.0f;
	float twiddle_host_ms = 0.0f;
	float garner_host_ms = 0.0f;
	float total_ms = 0.0f;
};

struct SetupUploadTiming {
	float twiddle_upload_ms = 0.0f;
	float mod_constants_ms = 0.0f;
	float garner_upload_ms = 0.0f;
	float total_ms = 0.0f;
};

struct NTTTiming {
	float ingress_fwd_ms = 0.0f;
	float h2d_ms = 0.0f;
	float fwd_pad_ntt_ms = 0.0f;
	float fwd_pad_ntt_a_ms = 0.0f;
	float fwd_pad_ntt_b_ms = 0.0f;
	float pointwise_mul_ms = 0.0f;
	float intt_ms = 0.0f;
	float crt_ms = 0.0f;
	float carry_ms = 0.0f;
	float d2h_ms = 0.0f;
	float total_ms = 0.0f;
};

NTTPrecomputed precompute_ntt(size_t N, PrecomputeTiming* timing_out = nullptr);
void upload_ntt_precomputed(NTTPrecomputed& pre, SetupUploadTiming* timing_out = nullptr);
NTTContext allocate_ntt_context(const NTTPrecomputed &pre, size_t L_A, size_t L_B);

void execute_ntt_multiply(
	NTTContext &ctx,
	const InputLimbType* a_pinned,
	const InputLimbType* b_pinned,
	vector<OutputLimbType> &C_out,
	NTTTiming* timing_out = nullptr,
	vector<uint64_t>* crt_hi_out = nullptr,
	vector<uint64_t>* crt_lo_out = nullptr,
	vector<uint64_t>* crt_mid_out = nullptr,
	vector<uint32_t>* crt_hi32_out = nullptr
);

void cleanup_ntt_context(NTTContext &ctx);
void cleanup_ntt_precomputed(NTTPrecomputed &pre);
