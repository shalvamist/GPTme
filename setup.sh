#!/bin/bash

git clone https://github.com/ggerganov/llama.cpp
export LLAMA_CUBLAS=on

cd llama.cpp
make clean
make libllama.so

export LLAMA_CPP_LIB=./llama.cpp/libllama.so
cd ..

CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64 -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir --verbos
