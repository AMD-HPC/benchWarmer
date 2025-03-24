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

benchWarmer is a synthetic microbenchmark suite that measures vector pipe throughput of various operations on integer and floating-point datatypes. This tool is compatible with both AMD and NVIDIA architectures.

The motivation for this tool is to study fundamental vector instruction level throughput performance for both compute-bound and bandwidth-bound workloads using a high-level implementation in CUDA or HIP.

## Implementation

The tool isolates a given vector operation for a given datatype and performs that operation many times in succession. While the amount of memory loaded is fixed, the amount of compute can be adjusted at compile time, using the `nOps` flag. Thus, the user can manually change the arithmetic intensity (ratio of computations to memory accesses), causing the benchmark's performance to be limited by either its compute or its bandwidth capabilities.

An example of a vector addition kernel is shown below. Each value is read from memory once (`a[offset]` on line 18) and stored in registers. The add operation is then performed `nOps` times (line 16) for each value read from memory, making the arithmetic intensity `nOps / sizeof(T)`.

![add_kernel](https://github.com/user-attachments/assets/832597d8-2675-4775-a115-d66ff96e7c2c)

The memory buffer is initalized with random numbers to emulate real world data.

**Supported datatypes:**
- int8
- int16
- int32
- int64
- fp16
- fp32
- fp64

**Supported operations:**
- Add
- Multiply
- MultiplyAdd
- Divide
- Rsqrt
- ShiftLeft
- ShiftRight
- RotateLeft
- RotateRight
- Type conversions

## Build
For AMD architectures:

`make amd nOps=<# of ops>`

For NVIDIA architectures:

`make nv nOps=<# of ops>`

The optional `nOps` argument (1000 by default) controls how many operations that will be computed on each piece of data. See [Implementation](https://github.com/AMD-HPC/benchWarmer/edit/main/README.md#implementation) for information on arithmetic intensity.

## Run
`./benchWarmer-amd <args>`
or
`./benchWarmer-nv <args>`

Output will be in a CSV format and sent to the console. One can redirect the output to a file by appending `|& tee <filename.log>`.

```
Arguments:
 -h, show this help message and exit

 -a, run all tests (all datatypes, all operations)

 Datatype arguments (defaults to all):
  --int, run all integer tests
  --fp, run all FP tests

  --int8, run int8 tests
  --int16, run int16 tests
  --int32, run int32 tests
  --int64, run int64 tests

  --fp16, run fp16 tests
  --fp32, run fp32 tests
  --fp64, run fp64 tests

 Operation arguments (defaults to all):
  --add, run Add tests only
  --mul, run Multiply tests only
  --muladd, run MulAdd tests only
  --div, run Division tests only
  --rsqrt, run rsqrt tests only
  --shift, run Shift tests only
  --rotate, run Rotate tests only
  --convert, run Type Conversion tests only
```

