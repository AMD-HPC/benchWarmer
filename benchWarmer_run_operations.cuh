#ifndef BENCHWARMER_RUN_OPERATIONS_CUH
#define BENCHWARMER_RUN_OPERATIONS_CUH

#include "benchWarmer_run_operations_mfma.cuh"

// Simplified version without MFMA support for unknown architectures
// Shift is only run for integer types and rsq only for floating point types
template<class T>
static void run_operations_not_mfma(bool add, bool mul, bool muladd, bool div, bool rsq, bool shift, int grid_size, int block_size, int buffer_size, int experiments, GPUArch arch, const char* datatype_override = nullptr) {
	if(add) {
		run_operations_valu<T>(add_kernel<T,nOps>, "Add", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
	}
	if(mul) {
		run_operations_valu<T>(mul_kernel<T,nOps>, "Mul", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
	}
	if(muladd) {
		run_operations_valu<T>(muladd_kernel<T,nOps>, "MulAdd", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
	}
	if(div) {
		run_operations_valu<T>(div_kernel<T,nOps>, "Div", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
	}
	if constexpr (std::is_integral_v<T>) {
		if(shift) {
			run_operations_valu<T>(shiftleft_kernel<T,nOps>, "ShiftLeft", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
			run_operations_valu<T>(shiftright_kernel<T,nOps>, "ShiftRight", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
		}
	} else {
		if(rsq) {
			run_operations_valu<T>(rsqrt_kernel<T,nOps>, "Rsqrt", grid_size, block_size, buffer_size, experiments, arch, datatype_override);
		}
	}
}

// Run relevant tests for a datatype. MFMA is skipped when the architecture has
// no MFMA instruction for the datatype (including GPUArch::UNKNOWN and CUDA).
template<class T>
static void run_operations(bool add, bool mul, bool muladd, bool div, bool rsq, bool shift, bool mfma, int grid_size, int block_size, int buffer_size, int experiments, GPUArch arch, const char* datatype_override = nullptr) {
	run_operations_not_mfma<T>(add, mul, muladd, div, rsq, shift, grid_size, block_size, buffer_size, experiments, arch, datatype_override);
#if __HIPCC__
	if(mfma) {
		run_operations_mfma(datatype_name<T>(datatype_override), grid_size, block_size, buffer_size, experiments, arch);
	}
#else
	(void)mfma;
#endif
}

#endif
