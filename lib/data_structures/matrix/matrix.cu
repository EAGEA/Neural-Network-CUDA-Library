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
    size_t index = row * nb_cols + col;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        data1[index] += data2[index];
    }
}

__global__ void __kernel_subtract(float *data1, const float *data2,
                                  size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        data1[index] -= data2[index];
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

__global__ void __kernel_multiply(float *data, float f,
                                  size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        data[index] *= f;
    }
}

__global__ void __kernel_hadamard_product(float *v1, float *v2,
                                          size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        v1[index] *= v2[index];
    }
}

__global__ void __kernel_sum(float *data, size_t nb_rows, size_t nb_cols)
{
    __shared__ size_t length;

    // Perform a reduction on "data".
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;
    length = nb_rows * nb_cols;

    for (size_t size = length; size >= 1; size /= 2)
    {
        if (size != length && index < size)
        {
            data[index] += data[index + size];
        }

        __syncthreads();

        if (size % 2 != 0 && index == size - 1)
        {
            data[1] += data[index];
            size --;
        }
    }
}

__global__ void __kernel_transpose(float *data1, const float *data2,
                                   size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index_1 = row * nb_cols + col;
    size_t index_2 = col * nb_rows + row;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        data1[index_2] = data2[index_1];
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

void matrix_cuda::subtract(const dim3 &block_dims, const dim3 &thread_dims,
                           const matrix &m1, const matrix &m2)
{
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(m1, &device_data1);
    start_operation(m2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_subtract<<<block_dims, thread_dims>>>(
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

void matrix_cuda::multiply(const dim3 &block_dims, const dim3 &thread_dims,
                      const matrix &m, float f)
{
    float *device_data;

    // Prepare data on device.
    start_operation(m, &device_data);
    // Do computations with CUDA threads.
    __kernel_multiply<<<block_dims, thread_dims>>>(
            device_data, f,
            m.get_dimensions().first, m.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    end_operation(m, &device_data);
}

void matrix_cuda::hadamard_product(dim3 block_dims, dim3 thread_dims,
                                   const matrix &v1, const matrix &v2)
{
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    matrix_cuda::start_operation(v1, &device_data1);
    matrix_cuda::start_operation(v2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_hadamard_product<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            v1.get_dimensions().first, v1.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_cuda::end_operation(v1, &device_data1);
    matrix_cuda::end_operation(v2, &device_data2);
}

void matrix_cuda::sum(const dim3 &block_dims, const dim3 &thread_dims,
                      float *result, const matrix &m)
{
    float *device_data;
    // Prepare data on device.
    start_operation(m, &device_data);
    // Do computations with CUDA threads.
    __kernel_sum<<<block_dims, thread_dims, sizeof(size_t)>>>(
            device_data,
            m.get_dimensions().first, m.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    CUDA_CHECK(cudaMemcpy(result, device_data,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_data));
}

void matrix_cuda::transpose(const dim3 &block_dims, const dim3 &thread_dims,
                            const matrix &result, const matrix &m)
{
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(result, &device_data1);
    start_operation(m, &device_data2);
    // Do computations with CUDA threads.
    __kernel_transpose<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            m.get_dimensions().first, m.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    end_operation(result, &device_data1);
    end_operation(m, &device_data2);
}