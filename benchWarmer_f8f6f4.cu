// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_api.h"
#if __HIPCC__
#include "benchWarmer_run_operations_mfma.cuh"
#endif

// MI350 (gfx950) only: fp8_e5m2/fp6/fp6_e3m2/fp4 scale MFMA
// (v_mfma_f32_32x32x64_f8f6f4). Each datatype only exposes the MFMA test;
// there is no scalar add/mul/etc. path for these formats.
//
// AMD's MI350 ISA reference and sibling non-scale AMDGPU builtins (e.g.
// __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8) name the e5m2 FP8 format "bf8"
// and the e3m2 FP6 format "bf6". The integer indices we pass to the f8f6f4
// builtin (1 for fp8_e5m2, 3 for fp6_e3m2) match that AMD ordering. The public
// API exposes the OCP MX format names instead.

void run_fp8_e5m2_benchmarks(
		bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
#if __HIPCC__
	if (mfma) {
		run_operations_mfma("fp8_e5m2", grid_size, block_size, buffer_size, experiments, arch);
	}
#else
	(void)mfma;
	(void)grid_size;
	(void)block_size;
	(void)buffer_size;
	(void)experiments;
	(void)arch;
#endif
}

void run_fp6_benchmarks(
		bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
#if __HIPCC__
	if (mfma) {
		run_operations_mfma("fp6", grid_size, block_size, buffer_size, experiments, arch);
	}
#else
	(void)mfma;
	(void)grid_size;
	(void)block_size;
	(void)buffer_size;
	(void)experiments;
	(void)arch;
#endif
}

void run_fp6_e3m2_benchmarks(
		bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
#if __HIPCC__
	if (mfma) {
		run_operations_mfma("fp6_e3m2", grid_size, block_size, buffer_size, experiments, arch);
	}
#else
	(void)mfma;
	(void)grid_size;
	(void)block_size;
	(void)buffer_size;
	(void)experiments;
	(void)arch;
#endif
}

void run_fp4_benchmarks(
		bool mfma,
		int grid_size, int block_size, int buffer_size, int experiments,
		GPUArch arch)
{
#if __HIPCC__
	if (mfma) {
		run_operations_mfma("fp4", grid_size, block_size, buffer_size, experiments, arch);
	}
#else
	(void)mfma;
	(void)grid_size;
	(void)block_size;
	(void)buffer_size;
	(void)experiments;
	(void)arch;
#endif
}
