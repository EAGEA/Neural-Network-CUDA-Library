//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "matrix.h"


using namespace cudaNN;

#define TILE_WIDTH 32


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

__global__ void __kernel_tiled_multiply(float *result,
                                  const float *data1, const float *data2,
                                  size_t nb_rows_1, size_t nb_cols_1,
                                  size_t nb_rows_2, size_t nb_cols_2)
{
    __shared__ float subTileData1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileData2[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    size_t col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    size_t row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float sum = .0f;

    for(int m = 0; m < (nb_cols_1 - 1)/TILE_WIDTH + 1; m++)
    {
        if( row < nb_cols_1 && m * TILE_WIDTH + tx < nb_cols_1)
        {
            subTileData1[ty][tx] = data1[row * nb_cols_1 + m * TILE_WIDTH + tx];
        }
        else subTileData1[ty][tx] = 0;

        if( col < nb_cols_2 && m * TILE_WIDTH + ty < nb_cols_2)
        {
            subTileData2[ty][tx] = data2[(m * TILE_WIDTH + ty) * nb_cols_2 + col];
        }
        else subTileData2[ty][tx] = 0;

        __syncthreads();
        if(row < nb_cols_1 && col < nb_cols_2)
        {
            for (int k = 0; k < TILE_WIDTH; ++k) {
                sum += subTileData1[ty][k] * subTileData2[k][tx];
            }
        }
        __syncthreads();
    }
    if(row < nb_cols_1 && col < nb_cols_2) {
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

__global__ void __kernel_do_hadamard_product(float *v1, float *v2,
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

__global__ void __kernel_do_sum(float *data, float *result,
                                size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    if (index < nb_rows * nb_cols)
    {
        // Do a reduction to compute the sum.
        extern __shared__ float shared_sum[];
        // Copy into shared memory.
        shared_sum[threadIdx.x] = data[blockIdx.x * blockDim.x + threadIdx.x];

        __syncthreads();

        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
        }

        __syncthreads();

        // Retrieve and sum the sum computed by each block.
        if (threadIdx.x == 0)
        {
            atomicAdd(result, shared_sum[0]);
        }
    }
}

__global__ void __kernel_do_transpose(float *data1, const float *data2,
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


void matrix_parallel::start_operation(const matrix &m, float **device_data)
{
    // Allocate memory on device.
    CUDA_CHECK(cudaMalloc(device_data, m.get_length() * sizeof(float)));
    // Copy the matrix to this memory.
    CUDA_CHECK(cudaMemcpy(*device_data, m.get_data(),
                          m.get_length() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void matrix_parallel::end_operation(const matrix &m, float **device_data)
{
    // Retrieve data from the device to the host (matrix).
    CUDA_CHECK(cudaMemcpy(m.get_data(), *device_data,
                          m.get_length() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Free device memory.
    CUDA_CHECK(cudaFree(*device_data));
}

void matrix_parallel::add(const matrix &m1, const matrix &m2)
{
    // Execute on CPU (more performances).
    for (size_t i = 0; i < m1.get_length(); i ++)
    {
        m1.get_data()[i] += m2.get_data()[i];
    }
    /*
    auto cuda_dims = util::get_cuda_2dims(m1.get_dimensions());
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

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
     */
}

void matrix_parallel::subtract(const matrix &m1, const matrix &m2)
{
    // Execute on CPU (more performances).
    for (size_t i = 0; i < m1.get_length(); i ++)
    {
        m1.get_data()[i] += m2.get_data()[i];
    }
    /*
    auto cuda_dims = util::get_cuda_2dims(m1.get_dimensions());
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

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
     */
}

void matrix_parallel::multiply(const matrix &m,
                      const matrix &m1, const matrix &m2)
{
    auto cuda_dims = util::get_cuda_2dims(m.get_dimensions());
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_result;
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(m, &device_result);
    start_operation(m1, &device_data1);
    start_operation(m2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_tiled_multiply<<<block_dims, thread_dims,TILE_WIDTH * TILE_WIDTH * 2 * sizeof(float)>>>(
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

void matrix_parallel::multiply(const matrix &m, float f)
{
    for (size_t i = 0; i < m.get_length(); i ++)
    {
        m.get_data()[i] *= f;
    }
    /*
    auto cuda_dims = util::get_cuda_2dims(m.get_dimensions());
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

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
     */
}

void matrix_parallel::do_hadamard_product(const matrix &v1, const matrix &v2)
{
    for (size_t i = 0; i < v1.get_length(); i ++)
    {
        v1.get_data()[i] *= v2.get_data()[i];
    }
    /*
    auto cuda_dims = util::get_cuda_2dims(v1.get_dimensions());
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(v1, &device_data1);
    start_operation(v2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_do_hadamard_product<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            v1.get_dimensions().first, v1.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    end_operation(v1, &device_data1);
    end_operation(v2, &device_data2);
     */
}

void matrix_parallel::do_sum(float *result, const matrix &m)
{
    // We use a reduction that assumes that the data is contained
    // in an array of size 2^n. Therefore, we round up to the next
    // power of 2 our matrix array.
    auto ceil2 = util::ceil2(m.get_length());
    auto cuda_dims = util::get_cuda_1dims(
            std::pair<size_t, size_t>(ceil2, 1));
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;
    *result = 0.f;

    float *device_data;
    float *device_result;

    // Prepare data on device.
    // - Allocate memory on device (of size 2^n).
    CUDA_CHECK(cudaMalloc(&device_data, ceil2 * sizeof(float)));
    // - Copy the matrix to this memory.
    CUDA_CHECK(cudaMemcpy(device_data, m.get_data(),
                          m.get_length() * sizeof(float),
                          cudaMemcpyHostToDevice));
    // Allocate result.
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_result, result,
                          sizeof(float),
                          cudaMemcpyHostToDevice));
    // Do computations with CUDA threads.
    __kernel_do_sum<<<block_dims, thread_dims, ceil2 * sizeof(float)>>>(
            device_data, device_result,
            m.get_dimensions().first, m.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    CUDA_CHECK(cudaMemcpy(result, device_result,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_data));
    CUDA_CHECK(cudaFree(device_result));
}

void matrix_parallel::do_transpose(matrix &result, const matrix &m)
{
    auto cuda_dims = util::get_cuda_2dims(m.get_dimensions());
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    start_operation(result, &device_data1);
    start_operation(m, &device_data2);
    // Do computations with CUDA threads.
    __kernel_do_transpose<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            m.get_dimensions().first, m.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    end_operation(result, &device_data1);
    end_operation(m, &device_data2);
}