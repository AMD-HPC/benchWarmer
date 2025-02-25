amd: benchWarmer-amd_$(nOps)
nv: benchWarmer-nv_$(nOps)
nvidia: nv

# Set number of computes
nOps ?= 1000

%-nv: %.cu
		nvcc $^ -o $@ -O3 -D nOps=$(nOps)
%-amd_$(nOps): %.cu
		hipcc -std=c++11 $^ -o $@ -O3 -D nOps=$(nOps)
		
clean: 
	rm benchWarmer-*
