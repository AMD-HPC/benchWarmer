# Copyright (C) 2025 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

# Set number of computes
nOps ?= 10000

# Debug build: make DEBUG=1 hip    or   make debug-hip
# Debug build: make DEBUG=1 cuda    or   make debug-cuda
# Saves intermediates and emits debug symbols (host + device for nvcc).
DEBUG ?= 0

BENCHWARMER_SRCS := benchWarmer_main.cu \
	benchWarmer_int8.cu benchWarmer_int16.cu benchWarmer_int32.cu benchWarmer_int64.cu \
	benchWarmer_fp8.cu benchWarmer_fp16.cu benchWarmer_bf16.cu benchWarmer_fp32.cu benchWarmer_fp32x2.cu benchWarmer_fp64.cu \
	benchWarmer_f8f6f4.cu benchWarmer_mfma.cu

BASEFLAGS := --std=c++17 -O3 -D nOps=$(nOps)

ifeq ($(DEBUG),1)
	HIPFLAGS := $(BASEFLAGS) -g -ggdb --save-temps
	# -g host, -G device debug; -ggdb/--save-temps via host compiler; -keep retains .ptx/.cubin intermediates
	CUDAFLAGS := $(BASEFLAGS) -g -G -Xcompiler -ggdb -Xcompiler --save-temps -keep
else
	HIPFLAGS := $(BASEFLAGS)
	CUDAFLAGS := $(BASEFLAGS)
endif

OBJDIR_HIP := obj_hip
OBJDIR_CUDA := obj_cuda

OBJS_HIP := $(BENCHWARMER_SRCS:%.cu=$(OBJDIR_HIP)/%.o)
OBJS_CUDA := $(BENCHWARMER_SRCS:%.cu=$(OBJDIR_CUDA)/%.o)

hip: $(OBJS_HIP)
	hipcc $(HIPFLAGS) $(OBJS_HIP) -o benchWarmer

cuda: $(OBJS_CUDA)
	nvcc $(CUDAFLAGS) $(OBJS_CUDA) -o benchWarmer

$(OBJDIR_HIP)/%.o: %.cu
	@mkdir -p $(OBJDIR_HIP)
	hipcc $(HIPFLAGS) -c $< -o $@

$(OBJDIR_CUDA)/%.o: %.cu
	@mkdir -p $(OBJDIR_CUDA)
	nvcc $(CUDAFLAGS) -c $< -o $@

# Root-level spill files from DEBUG=1 (hipcc --save-temps, nvcc -keep); obj_* is removed above
# nvcc uses cudafe1.{cpp,ii,s,...}, *.fatbin.c per TU, benchWarmer-nv_dlink.* + link.ii at link with -keep
DEBUG_CLEAN_ROOT := $(wildcard \
	benchWarmer_*-hip-*.bc benchWarmer_*-hip-*.hipi benchWarmer_*-hip-*.o benchWarmer_*-hip-*.s \
	benchWarmer_*-hip-*.out benchWarmer_*-hip-*.out.resolution.txt benchWarmer_*-hip-*.hipfb \
	benchWarmer_*-host-*.bc benchWarmer_*-host-*.hipi benchWarmer_*-host-*.s benchWarmer_*-host-*.o \
	benchWarmer_*.cu-hip-* \
	benchWarmer_*.cudafe1.* \
	benchWarmer_*.fatbin.c benchWarmer_*.ptx benchWarmer_*.cubin benchWarmer_*.fatbin benchWarmer_*.gpu \
	benchWarmer_*.module_id benchWarmer_*.reg.c \
	benchWarmer-nv_dlink.* \
	link.ii link.s \
	benchWarmer_*.cu.cpp* benchWarmer_*.cpp*.ii)

debug-hip:
	$(MAKE) DEBUG=1 hip

debug-cuda:
	$(MAKE) DEBUG=1 cuda

clean:
	rm -rf $(OBJDIR_HIP) $(OBJDIR_CUDA) benchWarmer
	$(RM) $(DEBUG_CLEAN_ROOT)

.PHONY: hip cuda clean debug-hip debug-cuda
