//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "matrix.h"


using namespace cudaNN;


/**
 * Kernel functions.
 */


__global__ void __kernel_add(float *data1, const float *data2,
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

__global__ void __kernel_multiply(float *result,
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

        result[row * nb_cols_2 + col] = sum;
    }
}


/**
 * Wrappers for call on host.
 */


void matrix_cuda::start_operation(const matrix &m, float **device_data)
{
    // Allocate memory on device.
    CUDA_CHECK(cudaMalloc(device_data, m.get_length() * sizeof(float)));
    // Copy the matrix to this memory.
    CUDA_CHECK(cudaMemcpy(*device_data, m.get_data(),
                          m.get_length() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void matrix_cuda::end_operation(const matrix &m, float **device_data)
{
    // Retrieve data from the device to the host (matrix).
    CUDA_CHECK(cudaMemcpy(m.get_data(), *device_data,
                          m.get_length() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Free device memory.
    CUDA_CHECK(cudaFree(*device_data));
}

void matrix_cuda::add(const dim3 &block_dims, const dim3 &thread_dims,
                      const matrix &m1, const matrix &m2)
{
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(m1, &device_data1);
    start_operation(m2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_add<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            m1.get_dimensions().first, m1.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    end_operation(m1, &device_data1);
    end_operation(m2, &device_data2);
}

void matrix_cuda::multiply(const dim3 &block_dims, const dim3 &thread_dims,
                           const matrix &m,
                           const matrix &m1, const matrix &m2)
{
    float *device_result;
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(m, &device_result);
    start_operation(m1, &device_data1);
    start_operation(m2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_multiply<<<block_dims, thread_dims>>>(
            device_result,
            device_data1, device_data2,
            m1.get_dimensions().first, m1.get_dimensions().second,
            m2.get_dimensions().first, m2.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    end_operation(m, &device_result);
    end_operation(m1, &device_data1);
    end_operation(m2, &device_data2);
}