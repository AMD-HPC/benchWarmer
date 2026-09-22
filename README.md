# benchWarmer
 ```
██████╗░███████╗███╗░░██╗░██████╗██╗░░██╗
██╔══██╗██╔════╝██╔██╗██║██║░░░░░██║░░██║
██████╦╝█████╗░░██╔██╗██║██║░░░░░███████║
██╔══██╗██╔══╝░░██║╚████║██║░░╔█╗██╔══██║
██████╦╝███████╗██║░╚███║╚██████║██║░░██║
╚═════╝░╚══════╝╚═╝░░╚══╝░╚═════╝╚═╝░░╚═╝
██╗░░░░░░░██╗░█████╗░██████╗░███╗░░░███╗███████╗██████╗░
██║░░██╗░░██║██╔══██╗██╔══██╗████╗░████║██╔════╝██╔══██╗
╚██╗████╗██╔╝███████║██████╔╝██╔████╔██║█████╗░░██████╔╝
░████╔═████║░██╔══██║██╔══██╗██║╚██╔╝██║██╔══╝░░██╔══██╗
░╚██╔╝░╚██╔╝░██║░░██║██║░░██║██║░╚═╝░██║███████╗██║░░██║
░░╚═╝░░░╚═╝░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚══════╝╚═╝░░╚═╝
 ```

benchWarmer is a synthetic microbenchmark suite that measures vector and matrix pipe throughput of various operations on integer and floating-point datatypes. This tool is compatible with both AMD and NVIDIA architectures.

The motivation for this tool is to study fundamental instruction level throughput performance for both compute-bound and bandwidth-bound workloads using a high-level implementation in CUDA or HIP.

## Implementation

The tool isolates a given operation for a given datatype and performs that operation many times in succession. While the amount of memory loaded is fixed, the amount of compute can be adjusted at compile time, using the `nOps` flag. Thus, the user can manually change the arithmetic intensity (ratio of computations to memory accesses), causing the benchmark's performance to be limited by either its compute or its bandwidth capabilities.

The memory buffer is initalized with random numbers to emulate real world data.

**Supported datatypes:**
- int8
- int16
- int32
- int64
- fp8
- fp16
- bf16
- fp32
- fp32x2 (float2)
- fp64
- fp8_e5m2 (MFMA only, MI350/gfx950 via `v_mfma_f32_32x32x64_f8f6f4`; AMD calls this `bf8`)
- fp6 (e2m3, MFMA only, MI350/gfx950 via `v_mfma_f32_32x32x64_f8f6f4`)
- fp6_e3m2 (MFMA only, MI350/gfx950 via `v_mfma_f32_32x32x64_f8f6f4`; AMD calls this `bf6`)
- fp4 (MFMA only, MI350/gfx950 via `v_mfma_f32_32x32x64_f8f6f4`)

**Supported operations:**
- Add
- Multiply
- MultiplyAdd
- Divide
- Rsqrt
- ShiftLeft
- ShiftRight
- MFMA

## Build
For AMD architectures:

`make -j nOps=<# of ops>`

For NVIDIA architectures:

`make -j cuda nOps=<# of ops>`

The optional `nOps` argument (10000 by default) controls how many operations that will be computed on each piece of data. See [Implementation](https://github.com/AMD-HPC/benchWarmer/tree/main#implementation) for information on arithmetic intensity.

## Run
`./benchWarmer <args>`

Output will be in a CSV format and sent to the console. One can redirect the output to a file by appending `|& tee <filename.log>`.

```
Arguments:
 -h, show this help message and exit

 -a, run all tests (all datatypes, all operations)

 --no-random-buffer-init, set the data buffer to be all 0's, rather than randomized

 Datatype arguments (defaults to all):
  --int, run all integer tests
  --fp, run all FP tests

  --int8, run int8 tests
  --int16, run int16 tests
  --int32, run int32 tests
  --int64, run int64 tests

  --fp8, run fp8 tests
  --fp16, run fp16 tests
  --bf16, run bf16 tests
  --fp32, run fp32 tests
  --fp32x2, run float2 tests
  --fp64, run fp64 tests
  --fp8_e5m2, run fp8_e5m2 MFMA tests (MI350/gfx950 only; AMD calls this "bf8")
  --fp6, run fp6 (e2m3) MFMA tests (MI350/gfx950 only)
  --fp6_e3m2, run fp6_e3m2 MFMA tests (MI350/gfx950 only; AMD calls this "bf6")
  --fp4, run fp4 MFMA tests (MI350/gfx950 only)

 Operation arguments (defaults to all):
  --add, run Add tests only
  --mul, run Multiply tests only
  --muladd, run MulAdd tests only
  --div, run Division tests only
  --rsqrt, run rsqrt tests only
  --shift, run Shift tests only
  --mfma, run MFMA tests only
```
