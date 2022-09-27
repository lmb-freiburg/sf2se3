#!/bin/bash

echo "CUDATOOLKIT_DIR ${CUDATOOLKIT_DIR}"

export CUDA_ROOT="${CUDATOOLKIT_DIR}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDATOOLKIT_DIR}"
export PATH="${CUDATOOLKIT_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDATOOLKIT_DIR}/lib64:${LD_LIBRARY_PATH}"

# CUPTI
export LD_LIBRARY_PATH="${CUDATOOLKIT_DIR}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"