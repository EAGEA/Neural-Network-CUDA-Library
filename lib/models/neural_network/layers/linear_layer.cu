//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "linear_layer.h"


/**
 * Kernel functions.
 */


__global__ void __kernel_execute_activation_functions(activation_function_t activation_function,
                                                      float *inputs, float *outputs,
                                                      size_t nb_neurons)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is in the matrix dimensions.
    if (col < nb_neurons && row < 1)
    {
        outputs[row * nb_neurons + col] = activation_function(inputs[row * nb_neurons + col]);
    }
}

__device__ void __kernel_update_activation_functions(matrix errors)
{
    // Update the weights.

    // Update the biases.

}

__global__ void __kernel_backward_propagation(matrix errors)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Update the parameters of the activation functions.
    __kernel_update_activation_functions(errors); //TODO
}


/**
 * Wrappers.
 */



void __execute_activation_functions(dim3 block_dims, dim3 thread_dims,
                                    activation_function_t activation_function,
                                    float *inputs, float *outputs,
                                    size_t nb_neurons)
{
    __kernel_execute_activation_functions<<<block_dims, thread_dims>>>(
            activation_function, 
            inputs, outputs,
            nb_neurons);
}

void __backward_propagation(dim3 block_dims, dim3 thread_dims, matrix errors)
{
    __kernel_backward_propagation<<<block_dims, thread_dims>>>(errors);
}
