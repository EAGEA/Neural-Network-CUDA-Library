//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "matrix.h"


using namespace cudaNN;


/**
 * Kernel functions.
 */


__global__ void __kernel_add(float *data1, float *data2,
                             size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        data1[row * nb_cols + col] += data2[row * nb_cols + col];
    }
}

__global__ void __kernel_multiply(float *output,
                                  const float *data1, const float *data2,
                                  size_t nb_rows_1, size_t nb_cols_1,
                                  size_t nb_rows_2, size_t nb_cols_2)
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
 * Wrappers for call on host.
 */


void matrix_cuda::add(const dim3 &block_dims, const dim3 &thread_dims,
                      float *host_data1, float *host_data2,
                      const size_t nb_rows, const size_t nb_cols)
{
    size_t length = nb_rows * nb_cols;

    float *device_data1;
    float *device_data2;

    // Allocate memory on device.
    CUDA_CHECK(cudaMalloc(&device_data1, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_data2, length * sizeof(float)));
    // Copy to this device memory.
    CUDA_CHECK(cudaMemcpy(device_data1, host_data1,
                          length * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_data2, host_data2,
                          length * sizeof(float),
                          cudaMemcpyHostToDevice));
    // Do computations with CUDA threads.
    __kernel_add<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            nb_rows, nb_cols);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy back the memory to host.
    CUDA_CHECK(cudaMemcpy(host_data1, device_data1,
                          length * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_data2, device_data2,
                          length * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Free device memory.
    CUDA_CHECK(cudaFree(device_data1));
    CUDA_CHECK(cudaFree(device_data2));
}

void matrix_cuda::multiply(const dim3 &block_dims, const dim3 &thread_dims,
                           float *host_output,
                           const float *host_data1, const float *host_data2,
                           size_t nb_rows_1, size_t nb_cols_1,
                           size_t nb_rows_2, size_t nb_cols_2)
{
    size_t length = nb_rows_1 * nb_cols_2;
    size_t length_1 = nb_rows_1 * nb_cols_1;
    size_t length_2 = nb_rows_2 * nb_cols_2;

    float *device_output;
    float *device_data1;
    float *device_data2;

    // Allocate memory on device.
    CUDA_CHECK(cudaMalloc(&device_output, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_data1, length_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_data2, length_2 * sizeof(float)));
    // Copy to this device memory.
    CUDA_CHECK(cudaMemcpy(device_data1, host_data1,
                          length_1 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_data2, host_data2,
                          length_2 * sizeof(float),
                          cudaMemcpyHostToDevice));
    // Do computations with CUDA threads.
    __kernel_multiply<<<block_dims, thread_dims>>>(
            device_output,
            device_data1, device_data2,
            nb_rows_1, nb_cols_1,
            nb_rows_2, nb_cols_2);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy back the memory to host.
    CUDA_CHECK(cudaMemcpy(host_output, device_output,
                          length * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Free device memory.
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_data1));
    CUDA_CHECK(cudaFree(device_data2));
}