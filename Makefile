amd: benchWarmer-amd_$(GPU)_$(nOps)
nv: benchWarmer-nv_$(GPU)_$(nOps)
nvidia: nv

# Set number of computes
nOps ?= 1000

%-nv: %.cu
		nvcc $^ -o $@ -O3 -D nOps=$(nOps)
%-amd_$(GPU)_$(nOps): %.cu
		hipcc -std=c++11 $^ -o $@ -O3 -D nOps=$(nOps)
		
clean: 
	rm benchWarmer-*
