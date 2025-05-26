# Copyright (C) 2025 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be 
# found in the LICENSE file # or at https://opensource.org/licenses/MIT.
amd: benchWarmer-amd
nv: benchWarmer-nv
nvidia: nv

# Set number of computes
nOps ?= 1000

%-nv: %.cu
		nvcc --std=c++11 $^ -o $@ -O3 -D nOps=$(nOps)
%-amd: %.cu
		hipcc --std=c++11 $^ -o $@ -O3 -D nOps=$(nOps)
		
clean: 
	rm benchWarmer-*
