//
// Created by Emilien Aufauvre on 12/12/2021.
//

#include "activation_functions.h"


using namespace cudaNN;


/**
 * Kernel functions.
 */


__global__ void __kernel_linear(float *results, float *inputs,
                                size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[row * nb_cols + col] = inputs[row * nb_cols + col];
    }
}

__global__ void __kernel_binary_step(float *results, float *inputs,
                                     size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[row * nb_cols + col] = inputs[row * nb_cols + col] < 0.f ?
                                       0.f : 1.f;
    }
}

__global__ void __kernel_sigmoid(float *results, float *inputs,
                                 size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[row * nb_cols + col] = 1.f / (1.f + exp(-inputs[row * nb_cols + col]));
    }
}

__global__ void __kernel_relu(float *results, float *inputs,
                              size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[row * nb_cols + col] = fmax(0.f, inputs[row * nb_cols + col]);
    }
}


/**
 * Wrappers for call on host.
 */


void activation_functions_cuda::linear(dim3 block_dims, dim3 thread_dims,
                                       float *results, float *inputs,
                                       size_t nb_rows, size_t nb_cols)
{
    __kernel_linear<<<block_dims, thread_dims>>>(
            results, inputs,
            nb_rows, nb_cols);
}

void activation_functions_cuda::binary_step(dim3 block_dims, dim3 thread_dims,
                                            float *results, float *inputs,
                                            size_t nb_rows, size_t nb_cols)
{
    __kernel_binary_step<<<block_dims, thread_dims>>>(
            results, inputs,
            nb_rows, nb_cols);
}

void activation_functions_cuda::sigmoid(dim3 block_dims, dim3 thread_dims,
                                        float *results, float *inputs,
                                        size_t nb_rows, size_t nb_cols)
{
    __kernel_sigmoid<<<block_dims, thread_dims>>>(
            results, inputs,
            nb_rows, nb_cols);
}

void activation_functions_cuda::relu(dim3 block_dims, dim3 thread_dims,
                                     float *results, float *inputs,
                                     size_t nb_rows, size_t nb_cols)
{
    __kernel_relu<<<block_dims, thread_dims>>>(
            results, inputs,
            nb_rows, nb_cols);
}
