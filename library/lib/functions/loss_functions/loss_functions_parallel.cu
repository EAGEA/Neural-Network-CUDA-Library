//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "loss_functions.h"


using namespace cudaNN;


/**
 * Kernel functions.
 */


__global__ void __kernel_mean_squared_error(float *errors,
                                            float *predictions, float *labels,
                                            size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = std::pow(
                    labels[nb_cols * i + col] - predictions[nb_cols * i + col],
                    2.0f);
        }
    }
}

__global__ void __kernel_mean_squared_error_derivative(float *errors,
                                            float *predictions, float *labels,
                                            size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = -2.f * (labels[nb_cols * i + col] - predictions[nb_cols * i + col]);
        }
    }
}

__global__ void __kernel_mean_absolute_error(float *errors,
                                             float *predictions, float *labels,
                                             size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = std::abs(labels[nb_cols * i + col] - predictions[nb_cols * i + col]);
        }
    }
}

__global__ void __kernel_mean_absolute_error_derivative(float *errors,
                                             float *predictions, float *labels,
                                             size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = predictions[nb_cols * i + col] > labels[nb_cols * i + col] ?
                                        +1.f : -1.f;
        }
    }
}

__global__ void __kernel_mean_bias_error(float *errors,
                                         float *predictions, float *labels,
                                         size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = labels[nb_cols * i + col] - predictions[nb_cols * i + col];
        }
    }
}

__global__ void __kernel_mean_bias_error_derivative(float *errors,
                                         float *predictions, float *labels,
                                         size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = -1.f;
        }
    }
}

__global__ void __kernel_hinge_loss(float *errors,
                                    float *predictions, float *labels,
                                    size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = std::fmax(0.f,
                                                  1.f - labels[nb_cols * i + col] * predictions[nb_cols * i + col]);
        }
    }
}

__global__ void __kernel_hinge_loss_derivative(float *errors,
                                    float *predictions, float *labels,
                                    size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            // TODO check if correct definition.
            errors[nb_cols * i + col] = predictions[nb_cols * i + col] > 1.f ?
                                        0.f : -labels[nb_cols * i + col] * 1.f;
        }
    }
}

__global__ void __kernel_binary_cross_entropy_loss(float *errors,
                                                   float *predictions, float *labels,
                                                   size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = -(labels[nb_cols * i + col]
                                          * logf(predictions[nb_cols * i + col])
                                          + (1.f - labels[nb_cols * i + col])
                                            * logf(1.f - predictions[nb_cols * i + col]));
        }
    }
}

__global__ void __kernel_binary_cross_entropy_loss_derivative(float *errors,
                                                              float *predictions, float *labels,
                                                              size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = -(labels[nb_cols * i + col]
                                          / predictions[nb_cols * i + col]
                                          - (1.f - labels[nb_cols * i + col])
                                            / (1.f - predictions[nb_cols * i + col]));
        }
    }
}

__global__ void __kernel_cross_entropy_loss(float *errors, float *loss,
                                            float *predictions, float *labels,
                                            size_t nb_rows, size_t nb_cols)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nb_rows * nb_cols)
    {
        // Do a reduction to compute the loss.
        extern __shared__ float shared_loss[];
        // Copy into shared memory.
        shared_loss[threadIdx.x] = -(labels[index] * logf(predictions[index]));

        __syncthreads();

        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                shared_loss[threadIdx.x] += shared_loss[threadIdx.x + stride];
            }
        }

        __syncthreads();

        // Retrieve and sum the sum computed by each block.
        if (threadIdx.x == 0)
        {
            atomicAdd(loss, shared_loss[0]);
        }

        __syncthreads();

        // Compute the softmax using the previously computed sum.
        if (index < nb_rows * nb_cols)
        {
            errors[index] = loss[0];
        }
    }
}

__global__ void __kernel_cross_entropy_loss_derivative(float *errors,
                                                       float *predictions, float *labels,
                                                       size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is in the output dimensions.
    if (col < nb_cols)
    {
        for (size_t i = 0; i < nb_rows; i ++)
        {
            errors[nb_cols * i + col] = -(labels[nb_cols * i + col]
                                          / predictions[nb_cols * i + col])
                                        + ((1.f - labels[nb_cols * i + col])
                                           / (1.f - predictions[nb_cols * i + col]));
        }
    }
}

void __helper(const matrix &errors,
              const matrix &predictions, const matrix &labels,
              void (kernel)(float *errors, float *predictions, float *labels,
                            size_t nb_rows, size_t nb_cols))
{
    auto cuda_dims = util::get_cuda_1dims(
            std::pair<size_t, size_t>(1, predictions.get_dimensions().second));
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_data0;
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    matrix_parallel::start_operation(errors, &device_data0);
    matrix_parallel::start_operation(predictions, &device_data1);
    matrix_parallel::start_operation(labels, &device_data2);
    // Do computations with CUDA threads.
    kernel<<<block_dims, thread_dims>>>(
            device_data0,
            device_data1, device_data2,
            errors.get_dimensions().first, errors.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_parallel::end_operation(errors, &device_data0);
    matrix_parallel::end_operation(predictions, &device_data1);
    matrix_parallel::end_operation(labels, &device_data2);
}

void __helper_cross_entropy_loss(const matrix &errors,
                                 const matrix &predictions, const matrix &labels,
                                 void (kernel)(float *errors, float *loss, float *predictions, float *labels,
                                               size_t nb_rows, size_t nb_cols))
{
    // We use a reduction that assumes that the data is contained
    // in an array of size 2^n. Therefore, we round up to the next
    // power of 2 our matrix array.
    auto ceil2 = util::ceil2(predictions.get_length());
    auto cuda_dims = util::get_cuda_1dims(
            std::pair<size_t, size_t>(ceil2, 1));
    auto block_dims = cuda_dims.first;
    auto thread_dims = cuda_dims.second;

    float *device_data0;
    float *device_data1;
    float *device_data2;
    float *loss;
    float zero = 0.f;

    // Prepare data on device.
    matrix_parallel::start_operation(errors, &device_data0);
    // For the matrices on which we will do the reduction:
    // - Allocate memory on device (of size 2^n).
    CUDA_CHECK(cudaMalloc(&device_data1, ceil2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_data2, ceil2 * sizeof(float)));
    // - Copy the matrix to this memory.
    CUDA_CHECK(cudaMemcpy(device_data1, predictions.get_data(),
                          predictions.get_length() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_data2, labels.get_data(),
                          labels.get_length() * sizeof(float),
                          cudaMemcpyHostToDevice));
    // Allocate for the sum.
    CUDA_CHECK(cudaMalloc(&loss, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(loss, &zero,
                          sizeof(float),
                          cudaMemcpyHostToDevice));
    // Do computations with CUDA threads.
    kernel<<<block_dims, thread_dims, (ceil2 / block_dims.x) * sizeof(float)>>>(
            device_data0, loss,
            device_data1, device_data2,
            errors.get_dimensions().first, errors.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_parallel::end_operation(errors, &device_data0);
    matrix_parallel::end_operation(predictions, &device_data1);
    matrix_parallel::end_operation(labels, &device_data2);
    CUDA_CHECK(cudaFree(loss));
}


/**
 * Wrappers for call on host.
 */


void loss_functions_parallel::mean_squared_error(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_mean_squared_error);
}

void loss_functions_parallel::mean_squared_error_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_mean_squared_error_derivative);
}

void loss_functions_parallel::mean_absolute_error(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_mean_absolute_error);
}

void loss_functions_parallel::mean_absolute_error_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_mean_absolute_error_derivative);
}

void loss_functions_parallel::mean_bias_error(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_mean_bias_error);
}

void loss_functions_parallel::mean_bias_error_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_mean_bias_error_derivative);
}

void loss_functions_parallel::hinge_loss(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_hinge_loss);
}

void loss_functions_parallel::hinge_loss_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_hinge_loss_derivative);
}

void loss_functions_parallel::binary_cross_entropy_loss(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_binary_cross_entropy_loss);
}

void loss_functions_parallel::binary_cross_entropy_loss_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_binary_cross_entropy_loss_derivative);
}

void loss_functions_parallel::cross_entropy_loss(std::vector<matrix *> m)
{
    __helper_cross_entropy_loss(*m[0], *m[1], *m[2], __kernel_cross_entropy_loss);
}

void loss_functions_parallel::cross_entropy_loss_derivative(std::vector<matrix *> m)
{
    __helper(*m[0], *m[1], *m[2], __kernel_cross_entropy_loss_derivative);
}