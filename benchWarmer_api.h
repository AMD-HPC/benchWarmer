// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "benchWarmer_common.cuh"

void run_int8_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool shift,
	bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_int16_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool shift,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_int32_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool shift,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_int64_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool shift,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp8_benchmarks(
	bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp16_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool rsq, bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_bf16_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool rsq, bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp32_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool rsq, bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

// float2 VALU throughput via the float2 instantiations of the op kernels (no MFMA)
void run_fp32x2_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool rsq,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp64_benchmarks(
	bool add, bool mul, bool muladd, bool div, bool rsq, bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

// MI350 (gfx950) only: scale MFMA f8/f6/f4 datatypes. These only support the
// MFMA test (no scalar add/mul/etc.). AMD's MI350 ISA reference and sibling
// AMDGPU builtins (e.g. __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8) name the
// e5m2/e3m2 formats "bf8"/"bf6"; we use the OCP MX format names instead.
void run_fp8_e5m2_benchmarks(
	bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp6_benchmarks(
	bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp6_e3m2_benchmarks(
	bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);

void run_fp4_benchmarks(
	bool mfma,
	int grid_size, int block_size, int buffer_size, int experiments,
	GPUArch arch);
