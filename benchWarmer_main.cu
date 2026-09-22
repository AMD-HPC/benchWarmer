// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "benchWarmer_api.h"

bool g_random_buffer_init = true;

int main(int argc, char **argv)
{

	//CLI parsing
	bool i8 = false, i16 = false, i32 = false, i64 = false, fp8 = false, fp16 = false, bf16 = false, fp32 = false, fp32x2 = false, fp64 = false;
	bool fp8_e5m2 = false, fp6 = false, fp6_e3m2 = false, fp4 = false;
	bool add = false, mul = false, muladd = false, div = false, rsq = false, shift = false, mfma = false;
	int grid_size = DEFAULT_WORKGROUPS;
	int block_size = DEFAULT_WORKGROUP_SIZE;
	int buffer_size = DEFAULT_DATASET_SIZE;
	int experiments = DEFAULT_NUM_EXPERIMENTS;

	int c, option_index = 0;
	static struct option long_options[] = {
					{"h",       no_argument,   0,  'h' },
					{"a",       no_argument,   0,  'a' },
					{"int8",    no_argument,   0,  'g' },
					{"int16",   no_argument,   0,  'b' },
					{"int32",   no_argument,   0,  'c' },
					{"int64",   no_argument,   0,  'd' },
					{"fp8",     no_argument,   0,  'z' },
					{"fp16",    no_argument,   0,  'v' },
					{"bf16",    no_argument,   0,  'A' },
					{"fp32",    no_argument,   0,  'e' },
					{"fp32x2",  no_argument,   0,  OPT_FP32X2 },
					{"fp64",    no_argument,   0,  'f' },
					{"fp8_e5m2", no_argument,  0,  'B' },
					{"fp6",      no_argument,  0,  'C' },
					{"fp6_e3m2", no_argument,  0,  'D' },
					{"fp4",      no_argument,  0,  'E' },
					{"int",     no_argument,   0,  'i' },
					{"fp",      no_argument,   0,  'j' },
					{"add",     no_argument,   0,  'k' },
					{"mul",     no_argument,   0,  'l' },
					{"muladd",  no_argument,   0,  'm' },
					{"div",     no_argument,   0,  't' },
					{"rsqrt",   no_argument,   0,  'n' },
					{"shift",   no_argument,   0,  'o' },
					{"mfma",    no_argument,   0,  'y' },
					{"grid-size",   required_argument,  0,  's' },
					{"block-size",  required_argument,  0,  'u' },
					{"buffer-size", required_argument,  0,  'x' },
					{"experiments", required_argument,  0,  'w' },
					{"no-random-buffer-init", no_argument, 0, OPT_NO_RANDOM_BUFFER_INIT },
					{0,         0,             0,  0   }
			};
	while ((c = getopt_long_only(argc, argv, "agbcdzvefijklmnos:u:x:w:yABCDE", long_options, &option_index)) != -1){
		if (c == -1)
			break;

		switch (c)
		{
			case 'a':
				i8=1, i16=1, i32=1, i64=1;
				fp8=1, bf16=1, fp16=1, fp32=1, fp32x2=1, fp64=1;
				fp8_e5m2=1, fp6=1, fp6_e3m2=1, fp4=1;
				break;
			case 'g':
				i8 = 1;
				break;
			case 'b':
				i16 = 1;
				break;
			case 'c':
				i32 = 1;
				break;
			case 'd':
				i64 = 1;
				break;
			case 'z':
				fp8 = 1;
				break;
			case 'v':
				fp16 = 1;
				break;
			case 'A':
				bf16 = 1;
				break;
			case 'e':
				fp32 = 1;
				break;
			case OPT_FP32X2:
				fp32x2 = 1;
				break;
			case 'f':
				fp64 = 1;
				break;
			case 'B':
				fp8_e5m2 = 1;
				break;
			case 'C':
				fp6 = 1;
				break;
			case 'D':
				fp6_e3m2 = 1;
				break;
			case 'E':
				fp4 = 1;
				break;
			case 'i':
				i8=1, i16=1, i32=1, i64=1;
				break;
			case 'j':
				fp8=1, bf16=1, fp16=1, fp32=1, fp32x2=1, fp64=1;
				fp8_e5m2=1, fp6=1, fp6_e3m2=1, fp4=1;
				break;
			case 'k':
				add = 1;
				break;
			case 'l':
				mul = 1;
				break;
			case 't':
				div = 1;
				break;
			case 'm':
				muladd = 1;
				break;
			case 'n':
				rsq = 1;
				break;
			case 'o':
				shift = 1;
				break;
			case 'y':
				mfma = 1;
				break;
			case 's':
				grid_size = atoi(optarg);
				break;
			case 'u':
				block_size = atoi(optarg);
				break;
			case 'x':
				buffer_size = atoi(optarg);
				break;
			case 'w':
				experiments = atoi(optarg);
				break;
			case OPT_NO_RANDOM_BUFFER_INIT:
				g_random_buffer_init = false;
				break;
			case 'h':
			default:
				printf("Usage: ./benchWarmer <args>\n");
				printf("\nArguments:\n");
				printf(" -h, show this help message and exit\n");
				printf("\n -a, run all tests (all datatypes, all operations)\n");

				printf("\n Problem size arguments:");
				printf("\n  --buffer-size, set size of buffer in bytes (default %u)", DEFAULT_DATASET_SIZE);
				printf("\n  --experiments, set number of experiments, not including a warmup kernel (default %u)\n", DEFAULT_NUM_EXPERIMENTS);

				printf("\n Work distribution arguments:");
				printf("\n  --grid-size, set grid size (default %u)", DEFAULT_WORKGROUPS);
				printf("\n  --block-size, set block size (default %u)\n", DEFAULT_WORKGROUP_SIZE);

				printf("\n Buffer initialization:");
				printf("\n  --no-random-buffer-init, zero-fill GPU buffers (default: random init before warmup and each experiment)\n");

				printf("\n Datatype arguments (defaults to all):");
				printf("\n  --int, run all integer tests");
				printf("\n  --fp, run all FP tests\n");

				printf("\n  --int8, run int8 tests");
				printf("\n  --int16, run int16 tests");
				printf("\n  --int32, run int32 tests");
				printf("\n  --int64, run int64 tests\n");
				printf("\n  --fp8, run fp8 tests");
				printf("\n  --fp16, run fp16 tests");
				printf("\n  --bf16, run bf16 tests");
				printf("\n  --fp32, run fp32 tests");
				printf("\n  --fp32x2, run float2 (fp32x2) tests");
				printf("\n  --fp64, run fp64 tests");
				printf("\n  --fp8_e5m2, run fp8_e5m2 MFMA tests (MI350/gfx950 only; AMD calls this \"bf8\")");
				printf("\n  --fp6, run fp6 (e2m3) MFMA tests (MI350/gfx950 only)");
				printf("\n  --fp6_e3m2, run fp6_e3m2 MFMA tests (MI350/gfx950 only; AMD calls this \"bf6\")");
				printf("\n  --fp4, run fp4 MFMA tests (MI350/gfx950 only)\n");

				printf("\n Operation arguments (defaults to all):");
				printf("\n  --add, run Add tests only");
				printf("\n  --mul, run Multiply tests only");
				printf("\n  --muladd, run MulAdd tests only");
				printf("\n  --div, run Division tests only");
				printf("\n  --rsqrt, run rsqrt tests only");
				printf("\n  --shift, run Shift tests only");
				printf("\n  --mfma, run MFMA tests only");
				printf("\n\n");
				exit(1);
		}
	}
	// If no operation/datatype is selected, do all of them
	if (! (add || mul || muladd || div || rsq || shift || mfma)) {
		add=1, mul=1, muladd=1, div=1, rsq=1, shift=1, mfma=1;
	}
	if (! (i8 || i16 || i32 || i64 || fp8 || fp16 || bf16 || fp32 || fp32x2 || fp64
	       || fp8_e5m2 || fp6 || fp6_e3m2 || fp4)) {
		i8=1, i16=1, i32=1, i64=1, fp8=1, fp16=1, bf16=1, fp32=1, fp32x2=1, fp64=1;
		fp8_e5m2=1, fp6=1, fp6_e3m2=1, fp4=1;
	}

	// Print header for CSV file
	printf("Datatype, Operation, Throughput mean (GFlops/s), Throughput stdev (GFlops/s), Duration mean (ms), Total flops, Total bytes accessed, AI, Workgroups, Threads, Experiments\n");
	GPUArch arch = getGPUArchitecture();

	if (i8) {
		run_int8_benchmarks(add, mul, muladd, div, shift, mfma,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (i16) {
		run_int16_benchmarks(add, mul, muladd, div, shift,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (i32) {
		run_int32_benchmarks(add, mul, muladd, div, shift,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (i64) {
		run_int64_benchmarks(add, mul, muladd, div, shift,
				grid_size, block_size, buffer_size, experiments, arch);
	}

	if (fp4) {
		run_fp4_benchmarks(mfma, grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp6) {
		run_fp6_benchmarks(mfma, grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp6_e3m2) {
		run_fp6_e3m2_benchmarks(mfma, grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp8) {
		run_fp8_benchmarks(mfma, grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp8_e5m2) {
		run_fp8_e5m2_benchmarks(mfma, grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp16) {
		run_fp16_benchmarks(add, mul, muladd, div, rsq, mfma,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (bf16) {
		run_bf16_benchmarks(add, mul, muladd, div, rsq, mfma,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp32) {
		run_fp32_benchmarks(add, mul, muladd, div, rsq, mfma,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp32x2) {
		run_fp32x2_benchmarks(add, mul, muladd, div, rsq,
				grid_size, block_size, buffer_size, experiments, arch);
	}
	if (fp64) {
		run_fp64_benchmarks(add, mul, muladd, div, rsq, mfma,
				grid_size, block_size, buffer_size, experiments, arch);
	}
}
