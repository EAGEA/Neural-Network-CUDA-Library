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
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = inputs[index];
    }
}

__global__ void __kernel_linear_derivative(float *results, float *inputs,
                                           size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = 1.f;
    }
}

__global__ void __kernel_binary_step(float *results, float *inputs,
                                     size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = inputs[index] < 0.f ? 0.f : 1.f;
    }
}

__global__ void __kernel_binary_step_derivative(float *results, float *inputs,
                                     size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = inputs[index] < 0.f ? 0.f : 1.f;
    }
}

__global__ void __kernel_sigmoid(float *results, float *inputs,
                                 size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = 1.f / (1.f + expf(-inputs[index]));
    }
}

__global__ void __kernel_sigmoid_derivative(float *results, float *inputs,
                                 size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        float sigmoid = 1.f / (1.f + expf(-inputs[index]));
        results[index] = sigmoid * (1.f - sigmoid);
    }
}

__global__ void __kernel_relu(float *results, float *inputs,
                              size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = fmax(0.f, inputs[index]);
    }
}

__global__ void __kernel_relu_derivative(float *results, float *inputs,
                                         size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = inputs[index] > 0 ? 1.f : 0.f;
    }
}

__global__ void __kernel_tanh(float *results, float *inputs,
                              size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        results[index] = tanh(inputs[index]);
    }
}

__global__ void __kernel_tanh_derivative(float *results, float *inputs,
                                         size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = row * nb_cols + col;

    // Check if the thread is in the matrix dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        float tanh_ = tanh(inputs[index]);
        results[index] = 1.f - tanh_ * tanh_;
    }
}

void __helper(dim3 block_dims, dim3 thread_dims,
              const matrix &results, const matrix &inputs,
              void (kernel)(float *result, float *inputs, size_t nb_rows, size_t nb_cols))
{
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    matrix_cuda::start_operation(results, &device_data1);
    matrix_cuda::start_operation(inputs, &device_data2);
    // Do computations with CUDA threads.
    kernel<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            results.get_dimensions().first, results.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_cuda::end_operation(results, &device_data1);
    matrix_cuda::end_operation(inputs, &device_data2);
}


/**
 * Wrappers for call on host.
 */


void activation_functions_cuda::linear(dim3 block_dims, dim3 thread_dims,
                                       std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_linear);
}

void activation_functions_cuda::linear_derivative(dim3 block_dims, dim3 thread_dims,
                                                  std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_linear_derivative);
}

void activation_functions_cuda::binary_step(dim3 block_dims, dim3 thread_dims,
                                            std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_binary_step);
}

void activation_functions_cuda::binary_step_derivative(dim3 block_dims, dim3 thread_dims,
                                                       std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_binary_step_derivative);
}

void activation_functions_cuda::sigmoid(dim3 block_dims, dim3 thread_dims,
                                        std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_sigmoid);
}

void activation_functions_cuda::sigmoid_derivative(dim3 block_dims, dim3 thread_dims,
                                                   std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_sigmoid_derivative);
}

void activation_functions_cuda::relu(dim3 block_dims, dim3 thread_dims,
                                     std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_relu);
}

void activation_functions_cuda::relu_derivative(dim3 block_dims, dim3 thread_dims,
                                                std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_relu_derivative);
}

void activation_functions_cuda::tanh(dim3 block_dims, dim3 thread_dims,
                                     std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_tanh);
}

void activation_functions_cuda::tanh_derivative(dim3 block_dims, dim3 thread_dims,
                                                std::vector<matrix *> m)
{
    __helper(block_dims, thread_dims,
             *m[0], *m[1],
             __kernel_tanh_derivative);
}
