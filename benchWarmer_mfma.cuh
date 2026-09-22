#ifndef BENCHWARMER_MFMA_CUH
#define BENCHWARMER_MFMA_CUH

#include "benchWarmer_common.cuh"

#if __HIPCC__
// MFMA kernels (AMD-specific) are defined in benchWarmer_mfma.cu, one per
// instruction. Each takes a dummy output buffer so the result is not optimized out.
using mfma_kernel_t = void (*)(float *);

// Returns the MFMA kernel for a datatype on the given architecture, or nullptr
// if that architecture has no MFMA instruction for the datatype
mfma_kernel_t select_mfma_kernel(const std::string &datatype, GPUArch arch);
#endif // __HIPCC__

#endif // BENCHWARMER_MFMA_CUH
