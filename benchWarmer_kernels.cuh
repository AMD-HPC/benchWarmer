#ifndef BENCHWARMER_KERNELS_CUH
#define BENCHWARMER_KERNELS_CUH

#include "benchWarmer_common.cuh"
#include <type_traits>

// Integer and FP8 Add/Mul use a two-write loop body; with a single write the
// compiler optimizes out the work for these datatypes.
template<class T>
__host__ __device__ constexpr bool use_unrolled_add_mul() {
#if __HIPCC__
	if constexpr (std::is_same_v<T, __hip_fp8_storage_t>)
		return true;
#endif
	return std::is_integral_v<T>;
}

template<typename T, int n>
__global__ void add_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];
	T x = a[0];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		if constexpr (use_unrolled_add_mul<T>())
		{
			// Unroll to remove extra loop overhead so operation counts are accurate
			#pragma unroll
			for(int i = 0; i + 1 < n; i+=2)
			{
				// Global writes to force the compiler to complete every operation
				x = a[offset] + x;
				a[offset] = a[offset] + x;
			}

			if constexpr ((n & 1) != 0)
			{
				x = a[offset] + x;
			}
		}
		else
		{
			// Unroll to remove extra loop overhead so operation counts are accurate
			#pragma unroll
			for(int i = 0; i < n; i++)
			{
				if constexpr (std::is_same_v<T, float2>)
					a[offset] = make_float2(a[offset].x + x.x, a[offset].y + x.y);
				else
					a[offset] = a[offset] + x;
			}
		}
	}
}

template<typename T, int n>
__global__ void mul_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];
	T x = a[0];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		if constexpr (use_unrolled_add_mul<T>())
		{
			// Unroll to remove extra loop overhead so operation counts are accurate
			#pragma unroll
			for(int i = 0; i + 1 < n; i+=2)
			{
				// Global writes to force the compiler to complete every operation
				x = a[offset] * x;
				a[offset] = a[offset] * x;
			}

			if constexpr ((n & 1) != 0)
			{
				x = a[offset] * x;
			}
		}
		else
		{
			// Unroll to remove extra loop overhead so operation counts are accurate
			#pragma unroll
			for(int i = 0; i < n; i++)
			{
				if constexpr (std::is_same_v<T, float2>)
					a[offset] = make_float2(a[offset].x * x.x, a[offset].y * x.y);
				else
					a[offset] = a[offset] * x;
			}
		}
	}
}

template<typename T, int n>
__global__ void muladd_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];
	T x = a[0];
	T y = (gid + 1 < nSize) ? a[1] : buf[0];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		// Unroll to remove extra loop overhead so operation counts are accurate
		#pragma unroll
		for(int i = 0; i < n; i++)
		{
			if constexpr (std::is_same_v<T, float2>)
				a[offset] = make_float2(a[offset].x * x.x + y.x, a[offset].y * x.y + y.y);
			else
				a[offset] = a[offset] * x + y;
		}
	}
}

template<typename T, int n>
__global__ void div_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];
	T x = a[0];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		// Unroll to remove extra loop overhead so operation counts are accurate
		#pragma unroll
		for(int i = 0; i < n; i++)
		{
			if constexpr (std::is_same_v<T, float2>)
				a[offset] = make_float2(a[offset].x / x.x, a[offset].y / x.y);
			else
				a[offset] = a[offset] / x;
		}
	}
}

template<typename T, int n>
__global__ void rsqrt_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		// Unroll to remove extra loop overhead so operation counts are accurate
		#pragma unroll
		for(int i = 0; i < n; i++)
		{
#if __HIPCC__
			if constexpr (std::is_same_v<T, float2>)
			{
				a[offset] = make_float2(__builtin_amdgcn_rsqf(a[offset].x), __builtin_amdgcn_rsqf(a[offset].y));
			}
			else if constexpr (std::is_same_v<T, double>)
			{
				a[offset] = __builtin_amdgcn_rsq(a[offset]);
			}
			else if constexpr (std::is_same_v<T, __half>)
			{
				_Float16 h;
				__builtin_memcpy(&h, &a[offset], sizeof(__half));
				h = __builtin_amdgcn_rsqh(h);
				__builtin_memcpy(&a[offset], &h, sizeof(__half));
			}
			else
			{
				a[offset] = (T)__builtin_amdgcn_rsqf((float)a[offset]);
			}
#else
			if constexpr (std::is_same_v<T, float2>)
			{
				a[offset] = make_float2(rsqrtf(a[offset].x), rsqrtf(a[offset].y));
			}
			else if constexpr (std::is_same_v<T, double>)
			{
				// No native FP64 rsqrt instruction on NVIDIA; this is a multi-instruction sequence
				a[offset] = rsqrt(a[offset]);
			}
			else if constexpr (std::is_same_v<T, __half>)
			{
				a[offset] = hrsqrt(a[offset]);
			}
			else
			{
				a[offset] = (T)rsqrtf((float)a[offset]);
			}
#endif
		}
	}
}

template<typename T, int n>
__global__ void shiftleft_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];
	T x = a[0];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		// Unroll to remove extra loop overhead so operation counts are accurate
		#pragma unroll
		for(int i = 0; i < n; i++)
		{
			a[offset] = a[offset] << x;
		}
	}
}

template<typename T, int n>
__global__ void shiftright_kernel(T *buf, uint32_t nSize)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nThreads  = gridDim.x * blockDim.x;

		if (gid >= nSize) return;

	T *a;
	a = &buf[gid];
	T x = a[0];

	for(uint32_t offset=0; offset + gid < nSize; offset += nThreads)
	{
		// Unroll to remove extra loop overhead so operation counts are accurate
		#pragma unroll
		for(int i = 0; i < n; i++)
		{
			a[offset] = a[offset] >> x;
		}
	}
}

// Kernel to initialize the buffer with random values
template <class T>
__global__ void initializeRandom(T *buffer, int nSize, unsigned long long seed) {
		const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
		if (gid < nSize) {
			hiprandState state;
			hiprand_init(seed + gid, 0, 0, &state); // Initialize the state with unique seed

#if __HIPCC__ || __CUDACC__
			if constexpr (std::is_same_v<T, float2>) {
				float u = hiprand_uniform(&state) * 100.f;
				buffer[gid] = float2{u, u};
			} else
#endif
			if constexpr (std::is_integral_v<T>) {
				buffer[gid] = static_cast<T>(hiprand(&state) % 100);  // [0, 99]
			} else {
				buffer[gid] = static_cast<T>(hiprand_uniform(&state) * 100);
			}
		}
}

#endif // BENCHWARMER_KERNELS_CUH
