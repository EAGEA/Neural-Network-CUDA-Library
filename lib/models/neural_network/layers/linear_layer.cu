//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "linear_layer.h"


using namespace cudaNN;


/**
 * Kernel functions.
 */


__device__ void __kernel_update_activation_functions(float *errors)
{
    // Update the weights.

    // Update the biases.

}

__global__ void __kernel_backward_propagation(float *errors)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Update the parameters of the activation functions.
    __kernel_update_activation_functions(errors); //TODO
}


/**
 * Wrappers for call on host.
 */


void linear_layer_cuda::backward_propagation(dim3 block_dims, dim3 thread_dims, float *errors)
{
    __kernel_backward_propagation<<<block_dims, thread_dims>>>(errors);
}
