//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "layer.h"


using namespace cudaNN;


/**
 * Kernel functions for backward propagation
 */


/**
* Error computation for the current layer
* @param weights - Weights of the previous layer
* @param d_previous_layer - Error of the previous layer
* @param d_current_layer - Error of the current layer (output to update)
* @param dim_weights - Dimension of the column of weights.
* @param dim_previous_layer - Dimension of the previous layer.
*/

__global__ void __kernel_layer_error(float *weights, float *d_previous_Layer,
                                     float *d_current_layer,
                                     int dim_weights, int dim_previous_layer)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0.0f;

    /*
    Each thread will contain one element of d_previous_layer and an entire column of weights
    if(TO DO) //condition to not be outside the limits of threads
    {
        for(TO DO)
        {
            value += weights[TO DO] * d_previous_layer[TO DO];
        }
        //ADD DERIVATIVE OF F(X)
        d_current_layer[TO DO] = value;
    }

     */
}


/**
* Update the weights of the current layer
* @param weights - Weights of the current layer (output to update)
* @param previous_layer - Error of the previous layer
* @param d_current_layer - Error of the current layer
* @param learning_rate
*/

// Without BATCH LEARNING ( no sum to do, no average) : ONLINE
__global__ void __kernel_update_weights(float *weights, float *previous_layer,
                                        float *d_current_layer, float learning_rate)
{
    /*
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    //Weight errors
    auto dwij = previous_layer[TO DO] * d_current_layer[TO DO]

    //Update weight
    weights[TO DO] = weights[TO DO] - learning_rate * dwij

    //(3) Synchronization before next layer
    __syncThreads();
    */
}

/**
* Update the biases of the current layer
* @param biases - Biases of the current layer (output to update)
* @param previous_layer - Error of the previous layer
* @param d_current_layer - Error of the current layer
* @param learning_rate
*/

// Without BATCH LEARNING ( no sum to do, no average) : ONLINE
__global__ void __kernel_update_biases(float *biases, float *previous_layer,
                                       float *d_current_layer, float learning_rate)
{
    /*
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    //Biases errors
    dbij = previous_layer[TO DO]

    //Update weight
    biases[TO DO] = biases[TO DO] - learning_rate * dbij

    //(3) Synchronization before next layer
    __syncThreads();
    */
}
/*
__global__ void __kernel_backward_propagation(float *errors)
{
    /*size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Update the functions of the activation functions.
    //__kernel_layer_error(errors,col,row); //TODO
    // Update the weights
    //__kernel_update_weights()
    // Update the biases
    //__kernel_update_biases()
}
*/

__global__ void __kernel_backward_propagation(float *v1, float *v2,
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


/**
 * Wrappers for call on host.
 */


void layer_cuda::backward_propagation(dim3 block_dims, dim3 thread_dims,
                                      const matrix &v1, const matrix &v2)
{
    float *device_data1;
    float *device_data2;

    // Prepare data on device.
    matrix_cuda::start_operation(v1, &device_data1);
    matrix_cuda::start_operation(v2, &device_data2);
    // Do computations with CUDA threads.
    __kernel_backward_propagation<<<block_dims, thread_dims>>>(
            device_data1, device_data2,
            v1.get_dimensions().first, v2.get_dimensions().second);
    // Wait for all threads.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Retrieve/free data from device.
    matrix_cuda::end_operation(v1, &device_data1);
    matrix_cuda::end_operation(v2, &device_data2);
}
