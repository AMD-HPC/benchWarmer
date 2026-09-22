// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_mfma.cuh"

#if __HIPCC__
// Vector types for MFMA operations (AMD-specific)
using int32_16vec = __attribute__((__vector_size__(16 * sizeof(int)))) int;
using int32_8vec = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using bf16_2vec = __attribute__((__vector_size__(1 * sizeof(__2i16))))  short;
using bf16_4vec = __attribute__((__vector_size__(2 * sizeof(__2i16))))  short;
using bf16_8vec = __attribute__((__vector_size__(4 * sizeof(__2i16))))  short;
using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using f16_2vec = __attribute__((__vector_size__(2 * sizeof(__2f16))))  float;
using f16_4vec = __attribute__((__vector_size__(4 * sizeof(__2f16))))  float;
using f64_4vec = __attribute__((__vector_size__(4 * sizeof(double)))) double;
using int32_1vec = __attribute__((__vector_size__(1 * sizeof(int)))) int;
using int32_4vec = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int64_1vec = __attribute__((__vector_size__(1 * sizeof(long)))) long;
using f32_1vec = __attribute__((__vector_size__(1 * sizeof(float)))) float;
using f64_1vec = __attribute__((__vector_size__(1 * sizeof(double)))) double;

// One device function per MFMA instruction. Each is compiled for every target
// architecture, so the #else branch is a placeholder for architectures that
// lack the instruction; select_mfma_kernel never launches it there.

// MI100/MI200 version (gfx908, gfx90a)
__device__ inline int32_16vec mfma_i8(int32_1vec a, int32_16vec result)
{
#if defined(__gfx908__) or defined(__gfx90a__)
	return __builtin_amdgcn_mfma_i32_32x32x8i8(a[0], a[0], result, 0, 0, 0);
#else
	return result;
#endif
}

// MI300 version (gfx940, gfx941, gfx942)
__device__ inline int32_16vec mfma_i8_mi300(int64_1vec a, int32_16vec result)
{
#if defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
	return __builtin_amdgcn_mfma_i32_32x32x16_i8(a[0], a[0], result, 0, 0, 0);
#else
	return result;
#endif
}

// MI350 version (gfx950)
__device__ inline int32_16vec mfma_i8_mi350(int32_4vec a, int32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_i32_32x32x32_i8(a, a, result, 0, 0, 0);
#else
	return result;
#endif
}

// MI300 only - FP8 support (gfx950 uses mfma_f8f6f4_fp8 instead)
__device__ inline f32_16vec mfma_f8(int64_1vec a, f32_16vec result)
{
#if defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__) or defined(__gfx950__)
	return __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a[0], a[0], result, 0, 0, 0);
#else
	return result;
#endif
}

// MI350 only (gfx950) - scale MFMA f8/f6/f4: 32x32x64, 131072 ops per instruction
// The builtin takes plain integer datatype indices; AMD's MI350 ISA reference
// (CBSZ/BLGP modifier tables for v_mfma_f32_32x32x64_f8f6f4) and the
// sibling non-scale AMDGPU builtins (e.g. __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8)
// name the corresponding formats fp8/bf8/fp6/bf6/fp4. Mapping to the OCP MX
// format names used in this codebase:
//   0 = fp8       (e4m3,  AMD ISA name "FP8")
//   1 = fp8_e5m2  (e5m2,  AMD ISA name "BF8")
//   2 = fp6       (e2m3,  AMD ISA name "FP6")
//   3 = fp6_e3m2  (e3m2,  AMD ISA name "BF6")
//   4 = fp4       (e2m1,  AMD ISA name "FP4")
__device__ inline f32_16vec mfma_f8f6f4_fp8(int32_8vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, result, 0, 0, 0, 0, 0, 0);
#else
	return result;
#endif
}

__device__ inline f32_16vec mfma_f8f6f4_fp8_e5m2(int32_8vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, result, 1, 1, 0, 0, 0, 0);
#else
	return result;
#endif
}

__device__ inline f32_16vec mfma_f8f6f4_fp6(int32_8vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, result, 2, 2, 0, 0, 0, 0);
#else
	return result;
#endif
}

__device__ inline f32_16vec mfma_f8f6f4_fp6_e3m2(int32_8vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, result, 3, 3, 0, 0, 0, 0);
#else
	return result;
#endif
}

__device__ inline f32_16vec mfma_f8f6f4_fp4(int32_8vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, a, result, 4, 4, 0, 0, 0, 0);
#else
	return result;
#endif
}

// MI100 version (gfx908); gfx90a uses mfma_bf16_mi200
__device__ inline f32_16vec mfma_bf16(bf16_2vec a, f32_16vec result)
{
#if defined(__gfx908__)
	return __builtin_amdgcn_mfma_f32_32x32x4bf16(a, a, result, 0, 0, 0);
#else
	return result;
#endif
}

// MI200/MI300 version
__device__ inline f32_16vec mfma_bf16_mi200(bf16_4vec a, f32_16vec result)
{
#if defined(__gfx90a__) or defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
	return __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, a, result, 0, 0, 0);
#else
	return result;
#endif
}

// MI350 version (gfx950)
__device__ inline f32_16vec mfma_bf16_mi350(bf16_8vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, a, result, 0, 0, 0);
#else
	return result;
#endif
}

// Pre-MI350 version
__device__ inline f32_16vec mfma_f16(f16_2vec a, f32_16vec result)
{
#if defined(__gfx908__) or defined(__gfx90a__) or defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
	return __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, result, 0, 0, 0);
#else
	return result;
#endif
}

__device__ inline f32_16vec mfma_f16_mi350(f16_4vec a, f32_16vec result)
{
#if defined(__gfx950__)
	return __builtin_amdgcn_mfma_f32_32x32x16_f16(a, a, result, 0, 0, 0);
#else
	return result;
#endif
}

__device__ inline f32_16vec mfma_f32(f32_1vec a, f32_16vec result)
{
#if defined(__gfx908__) or defined(__gfx90a__) or defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__) or defined(__gfx950__)
	return __builtin_amdgcn_mfma_f32_32x32x2f32(a[0], a[0], result, 0, 0, 0);
#else
	return result;
#endif
}

// MI200 and MI300 - FP64 MFMA support
__device__ inline f64_4vec mfma_f64(f64_1vec a, f64_4vec result)
{
#if defined(__gfx90a__) or defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__) or defined(__gfx950__)
	return __builtin_amdgcn_mfma_f64_16x16x4f64(a[0], a[0], result, 0, 0, 0);
#else
	return result;
#endif
}

// Defines <step>_kernel, which issues nOps dependent calls to the MFMA device
// function <step>. Each MFMA's input is reinterpreted from the previous result.
#define DEFINE_MFMA_KERNEL(step, T_in, T_out)                              \
__global__ void step##_kernel(float *dummy)                                \
{                                                                          \
	const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;            \
                                                                           \
	T_out result = {0};                                                    \
                                                                           \
	for(int i = 0; i < nOps; ++i)                                          \
	{                                                                      \
		result = step(*reinterpret_cast<T_in*>(&result), result);          \
	}                                                                      \
                                                                           \
	/* Prevent compiler optimization by using the result */                \
	if (gid == 0) {                                                        \
		T_out *output = (T_out*)dummy;                                     \
		output[0] = result;                                                \
	}                                                                      \
}

DEFINE_MFMA_KERNEL(mfma_i8,              int32_1vec, int32_16vec)
DEFINE_MFMA_KERNEL(mfma_i8_mi300,        int64_1vec, int32_16vec)
DEFINE_MFMA_KERNEL(mfma_i8_mi350,        int32_4vec, int32_16vec)
DEFINE_MFMA_KERNEL(mfma_f8,              int64_1vec, f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f8f6f4_fp8,      int32_8vec, f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f8f6f4_fp8_e5m2, int32_8vec, f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f8f6f4_fp6,      int32_8vec, f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f8f6f4_fp6_e3m2, int32_8vec, f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f8f6f4_fp4,      int32_8vec, f32_16vec)
DEFINE_MFMA_KERNEL(mfma_bf16,            bf16_2vec,  f32_16vec)
DEFINE_MFMA_KERNEL(mfma_bf16_mi200,      bf16_4vec,  f32_16vec)
DEFINE_MFMA_KERNEL(mfma_bf16_mi350,      bf16_8vec,  f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f16,             f16_2vec,   f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f16_mi350,       f16_4vec,   f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f32,             f32_1vec,   f32_16vec)
DEFINE_MFMA_KERNEL(mfma_f64,             f64_1vec,   f64_4vec)

mfma_kernel_t select_mfma_kernel(const std::string &datatype, GPUArch arch)
{
	if (datatype == "int8") {
		switch (arch) {
			case GPUArch::GFX908:
			case GPUArch::GFX90A: return mfma_i8_kernel;
			case GPUArch::GFX940: return mfma_i8_mi300_kernel;
			case GPUArch::GFX950: return mfma_i8_mi350_kernel;
			default:              return nullptr;
		}
	}
	// gfx940 uses v_mfma_f32_32x32x16_fp8_fp8; gfx950 uses the higher-throughput
	// v_mfma_f32_32x32x64_f8f6f4 instruction with the fp8 datatype index
	if (datatype == "fp8") {
		switch (arch) {
			case GPUArch::GFX940: return mfma_f8_kernel;
			case GPUArch::GFX950: return mfma_f8f6f4_fp8_kernel;
			default:              return nullptr;
		}
	}
	if (datatype == "fp16") {
		switch (arch) {
			case GPUArch::GFX908:
			case GPUArch::GFX90A:
			case GPUArch::GFX940: return mfma_f16_kernel;
			case GPUArch::GFX950: return mfma_f16_mi350_kernel;
			default:              return nullptr;
		}
	}
	if (datatype == "bf16") {
		switch (arch) {
			case GPUArch::GFX908: return mfma_bf16_kernel;
			case GPUArch::GFX90A:
			case GPUArch::GFX940: return mfma_bf16_mi200_kernel;
			case GPUArch::GFX950: return mfma_bf16_mi350_kernel;
			default:              return nullptr;
		}
	}
	if (datatype == "fp32") {
		return arch == GPUArch::UNKNOWN ? nullptr : mfma_f32_kernel;
	}
	// gfx908 has no FP64 MFMA
	if (datatype == "fp64") {
		switch (arch) {
			case GPUArch::GFX90A:
			case GPUArch::GFX940:
			case GPUArch::GFX950: return mfma_f64_kernel;
			default:              return nullptr;
		}
	}
	// MI350 (gfx950) only: scale MFMA f8/f6/f4 datatypes
	if (arch == GPUArch::GFX950) {
		if (datatype == "fp8_e5m2") return mfma_f8f6f4_fp8_e5m2_kernel;
		if (datatype == "fp6")      return mfma_f8f6f4_fp6_kernel;
		if (datatype == "fp6_e3m2") return mfma_f8f6f4_fp6_e3m2_kernel;
		if (datatype == "fp4")      return mfma_f8f6f4_fp4_kernel;
	}
	return nullptr;
}
#endif // __HIPCC__
