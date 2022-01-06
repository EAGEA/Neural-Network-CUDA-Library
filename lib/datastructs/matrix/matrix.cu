//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "matrix.h"


using namespace cudaNN;


/**
 * Kernel functions.
 */


__global__ void __kernel_add(float *data1, float *data2,
                             const size_t nb_rows, const size_t nb_cols)
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
                                  const size_t &nb_rows_1, const size_t &nb_cols_1,
                                  const size_t &nb_rows_2, const size_t &nb_cols_2)
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


void matrix_cuda::allocate(const std::string &id,
                           const size_t length,
                           float **device_data)
{
    cudaError_t err = cudaMalloc(device_data, length * sizeof(float));

    if (err != cudaSuccess)
    {
        // Invalid.
        util::ERROR("matrix_cuda::allocate",
                    "matrix::_id = " + id + " >> memory allocation on device failed", err);
        util::ERROR_EXIT();
    }
}

void matrix_cuda::free(const std::string &id, float *device_data)
{
    cudaError_t err = cudaFree(device_data);

    if (err != cudaSuccess)
    {
        // Invalid.
        util::ERROR("matrix_cuda::free",
                    "matrix::_id = " + id + " >> memory deallocation on device failed", err);
        util::ERROR_EXIT();
    }
}

void matrix_cuda::copy_host_to_device(const std::string &id,
                                      float *host_data, float *device_data, size_t size)
{
    cudaError_t err = cudaMemcpy(device_data, host_data, size * sizeof(float),
                                 cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        // Invalid.
        util::ERROR("matrix_cuda::copy_host_to_device",
                    "matrix::_id = " + id + " >> copy on device failed", err);
        util::ERROR_EXIT();
    }
}

void matrix_cuda::copy_device_to_host(const std::string &id,
                                      float *host_data, float *device_data, size_t size)
{
    cudaError_t err = cudaMemcpy(host_data, device_data, size * sizeof(float),
                                 cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        // Invalid.
        util::ERROR("matrix_cuda::copy_device_to_host",
                    "matrix::_id = " + id + " >> copy on host failed", err);
        util::ERROR_EXIT();
    }
}

void matrix_cuda::add(const dim3 &block_dims, const dim3 &thread_dims,
                      float *data1, float *data2,
                      const size_t nb_rows, const size_t nb_cols)
{
    __kernel_add<<<block_dims, thread_dims>>>(
            data1, data2,
            nb_rows, nb_cols);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        // Invalid.
        util::ERROR("matrix_cuda::add",
                    "device synchronize failed", err);
        util::ERROR_EXIT();
    }
}

void matrix_cuda::multiply(const dim3 &block_dims, const dim3 &thread_dims,
                           float *output,
                           const float *data1, const float *data2,
                           const size_t &nb_rows_1, const size_t &nb_cols_1,
                           const size_t &nb_rows_2, const size_t &nb_cols_2)
{
    __kernel_multiply<<<block_dims, thread_dims>>>(
            output,
            data1, data2,
            nb_rows_1, nb_cols_1,
            nb_rows_2, nb_cols_2);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        // Invalid.
        util::ERROR("matrix_cuda::multiply",
                    "device synchronize failed", err);
        util::ERROR_EXIT();
    }
}