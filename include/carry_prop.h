// carry_prop.h
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"

__global__ void carry_intra_segment_kernel(
    const uint64_t*   __restrict__ C_hi,
    const uint64_t*   __restrict__ C_lo,
    OutputLimbType* __restrict__ out,
    int64_t*          __restrict__ seg_carry,
    size_t N);

__global__ void carry_inter_segment_kernel(
    int64_t* __restrict__ seg_carry,
    size_t   num_segs);

__global__ void carry_fixup_kernel(
    OutputLimbType* __restrict__ out,
    const int64_t*    __restrict__ seg_carry_in,
    int64_t*          __restrict__ seg_carry_out,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag);

#if defined(NATIVE_HOST_LIMBS)
__global__ void carry_intra_segment_kernel_u160(
    const uint64_t* __restrict__ C_lo,
    const uint64_t* __restrict__ C_mid,
    const uint32_t* __restrict__ C_hi,
    OutputLimbType* __restrict__ out,
    uint64_t*         __restrict__ seg_carry_lo,
    uint64_t*         __restrict__ seg_carry_mid,
    uint32_t*         __restrict__ seg_carry_hi,
    size_t N);

__global__ void carry_inter_segment_kernel_u160(
    uint64_t* __restrict__ seg_carry_lo,
    uint64_t* __restrict__ seg_carry_mid,
    uint32_t* __restrict__ seg_carry_hi,
    size_t   num_segs);

__global__ void carry_fixup_kernel_u160(
    OutputLimbType* __restrict__ out,
    const uint64_t*   __restrict__ seg_carry_in_lo,
    const uint64_t*   __restrict__ seg_carry_in_mid,
    const uint32_t*   __restrict__ seg_carry_in_hi,
    uint64_t*         __restrict__ seg_carry_out_lo,
    uint64_t*         __restrict__ seg_carry_out_mid,
    uint32_t*         __restrict__ seg_carry_out_hi,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag);
#endif