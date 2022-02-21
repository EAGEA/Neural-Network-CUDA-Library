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

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = inputs[nb_cols * i + col];
        }
    }
}

__global__ void __kernel_linear_derivative(float *results, float *inputs,
                                           size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = 1.f;
        }
    }
}

__global__ void __kernel_binary_step(float *results, float *inputs,
                                     size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = inputs[nb_cols * i + col] < 0.f ? 0.f : 1.f;
        }
    }
}

__global__ void __kernel_binary_step_derivative(float *results, float *inputs,
                                                size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = 0.f;
        }
    }
}

__global__ void __kernel_sigmoid(float *results, float *inputs,
                                 size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = 1.f / (1.f + expf(-inputs[nb_cols * i + col]));
        }
    }
}

__global__ void __kernel_sigmoid_derivative(float *results, float *inputs,
                                            size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            float sigmoid = 1.f / (1.f + expf(-inputs[nb_cols * i + col]));
            results[nb_cols * i + col] = sigmoid * (1.f - sigmoid);
        }
    }
}

__global__ void __kernel_relu(float *results, float *inputs,
                              size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = fmax(0.f, inputs[nb_cols * i + col]);
        }
    }
}

__global__ void __kernel_relu_derivative(float *results, float *inputs,
                                         size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = inputs[nb_cols * i + col] > 0 ? 1.f : 0.f;
        }
    }
}

__global__ void __kernel_tanh(float *results, float *inputs,
                              size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            results[nb_cols * i + col] = tanhf(inputs[nb_cols * i + col]);
        }
    }
}

__global__ void __kernel_tanh_derivative(float *results, float *inputs,
                                         size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            float tanh_ = tanhf(inputs[nb_cols * i + col]);
            results[nb_cols * i + col] = 1.f - tanh_ * tanh_;
        }
    }
}

__global__ void __kernel_softmax(float *results, float *inputs,
                                 float *sum, float max_d,
                                 size_t nb_rows, size_t nb_cols)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nb_rows * nb_cols)
    {
        // Do a reduction to compute the sum.
        extern __shared__ float shared_sum[];
        // Copy into shared memory.
        shared_sum[threadIdx.x] = expf(inputs[index] - max_d);

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
            atomicAdd(sum, shared_sum[0]);
        }

        __syncthreads();

        // Compute the softmax using the previously computed sum.
        if (index < nb_rows * nb_cols)
        {
            results[index] = expf(inputs[index] - max_d) / sum[0];
        }
    }
}

__global__ void __kernel_softmax_derivative(float *results, float *inputs,
                                            float *sum, float max_d,
                                            size_t nb_rows, size_t nb_cols)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nb_rows * nb_cols)
    {
        // Do a reduction to compute the sum.
        extern __shared__ float shared_sum[];
        // Copy into shared memory.
        shared_sum[threadIdx.x] = inputs[index];

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
            atomicAdd(sum, shared_sum[0]);
        }

        __syncthreads();

        // Compute the derivative using the previously computed sum.
        if (index < nb_rows * nb_cols)
        {
            size_t row = index / nb_cols;
            size_t col = index % nb_cols;
            float softmax_x = expf(inputs[row] - max_d) / sum[0];
            float softmax_y = expf(inputs[col] - max_d) / sum[0];

            if (row == col)
            {
                results[index] = softmax_x * (1 - softmax_x);
            }
            else
            {
                results[index] = -softmax_x * softmax_y;
            }
        }
    }
}

void __helper(const matrix &results, const matrix &inputs,
              void (kernel)(float *result, float *inputs, size_t nb_rows, size_t nb_cols))
{
    auto cuda_dims = util::get_cuda_1dims(
            std::pair<size_t, size_t>(1, inputs.get_dimensions().second));
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    matrix_parallel::start_operation(results, &device_data1);
    matrix_parallel::start_operation(inputs, &device_data2);

    // Do computations with CUDA threads.
    kernel<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            results.get_dimensions().first, results.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_parallel::end_operation(results, &device_data1);
    matrix_parallel::end_operation(inputs, &device_data2);
}

void __helper_softmax(const matrix &results, const matrix &inputs, float max_data,
                      void (kernel)(float *result, float *inputs, float *sum, float max_data,
                                    size_t nb_rows, size_t nb_cols))
{
    // We use a reduction that assumes that the data is contained
    // in an array of size 2^n. Therefore, we round up to the next
    // power of 2 our matrix array.
    auto ceil2 = util::ceil2(inputs.get_length());
    auto cuda_dims = util::get_cuda_1dims(
            std::pair<size_t, size_t>(ceil2, 1));
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_data1;
    float *device_data2;
    float *sum;
    float zero = 0.f;

    // Prepare data on device.
    matrix_parallel::start_operation(results, &device_data1);
    // For the matrix on which we will do the reduction:
    // - Allocate memory on device (of size 2^n).
    CUDA_CHECK(cudaMalloc(&device_data2, ceil2 * sizeof(float)));
    // - Copy the matrix to this memory.
    CUDA_CHECK(cudaMemcpy(device_data2, inputs.get_data(),
                          inputs.get_length() * sizeof(float),
                          cudaMemcpyHostToDevice));
    // Allocate for the sum.
    CUDA_CHECK(cudaMalloc(&sum, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(sum, &zero,
                          sizeof(float),
                          cudaMemcpyHostToDevice));
    // Do computations with CUDA threads.
    kernel<<<block_dims, thread_dims, (ceil2 / block_dims.x) * sizeof(float)>>>(
            device_data1, device_data2,
            sum, max_data,
            results.get_dimensions().first, results.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_parallel::end_operation(results, &device_data1);
    matrix_parallel::end_operation(inputs, &device_data2);
    CUDA_CHECK(cudaFree(sum));
}


/**
 * Wrappers for call on host.
 */


void activation_functions_parallel::linear(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_linear);
}

void activation_functions_parallel::linear_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_linear_derivative);
}

void activation_functions_parallel::binary_step(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_binary_step);
}

void activation_functions_parallel::binary_step_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_binary_step_derivative);
}

void activation_functions_parallel::sigmoid(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_sigmoid);
}

void activation_functions_parallel::sigmoid_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_sigmoid_derivative);
}

void activation_functions_parallel::relu(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_relu);
}

void activation_functions_parallel::relu_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_relu_derivative);
}

void activation_functions_parallel::tanh(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_tanh);
}

void activation_functions_parallel::tanh_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1],__kernel_tanh_derivative);
}

void activation_functions_parallel::softmax(std::vector<matrix *> m)
{
    __helper_softmax(*m[0], *m[1],m[1]->get_max(),__kernel_softmax);
}

void activation_functions_parallel::softmax_derivative(std::vector<matrix *> m)
{
    __helper_softmax(*m[0], *m[1],m[1]->get_max(),__kernel_softmax_derivative);
}