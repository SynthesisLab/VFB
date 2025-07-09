all: ltl_nocache

ltl_nocache: ltl_nocache.cu json.hpp
	nvcc $< -o $@ -arch=sm_80 -std=c++20
