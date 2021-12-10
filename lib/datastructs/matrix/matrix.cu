//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "matrix.h"


/**
 * Kernel functions.
 */


__global__ void __kernel_add(float *output, 
                             const float *data1, const float *data2,
                             const size_t nb_rows, const size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        float sum = 0.f;

        sum += data1[row * nb_cols + col];
        sum += data2[row * nb_cols + col];

        output[row * nb_cols + col] = sum;
    }
}

__global__ void __kernel_multiply(float *output, 
                                  const float *data1, const float *data2,
                                  const size_t nb_rows_1, const size_t nb_cols_1,
                                  const size_t nb_rows_2, const size_t nb_cols_2)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows_1 && col < nb_cols_2)
    {
        float sum = .0f;

        for (size_t i = 0; i < nb_cols_1; i ++)
        {
            sum += data1[row * nb_cols_1 + i] * data2[i * nb_cols_2 + col];
        }

        output[row * nb_cols_2 + col] = sum;
    }
}


/**
 * Wrappers.
 */


void __allocate(const std::pair<size_t, size_t> dimensions, float *&device_data)
{
    cudaError_t err = cudaMalloc(&device_data, dimensions.first * dimensions.second * sizeof(float));

    if (err == cudaErrorMemoryAllocation)
    {
        // Invalid.
        util::ERROR("matrix::allocate", "memory allocation on device failed");
        util::ERROR_EXIT();
    }
}

void __free(float *&device_data)
{
    cudaFree(device_data);
}

void __add(const dim3 block_dims, const dim3 thread_dims,
           float *output, 
           const float *data1, const float *data2,
           const size_t nb_rows, const size_t nb_cols)
{
    __kernel_add<<<block_dims, thread_dims>>>(
            output,
            data1, data2,
            nb_rows, nb_cols);
}

void __multiply(const dim3 block_dims, const dim3 thread_dims,
                float *output, 
                const float *data1, const float *data2,
                const size_t nb_rows_1, const size_t nb_cols_1,
                const size_t nb_rows_2, const size_t nb_cols_2)
{
    __kernel_multiply<<<block_dims, thread_dims>>>(
            output, 
            data1, data2, 
            nb_rows_1, nb_cols_1, 
            nb_rows_2, nb_cols_2);
}

