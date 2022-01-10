//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "layer.h"


using namespace cudaNN;


/**
 * Kernel functions for backward propagation
 */

__global__ void __kernel_layer_error(float *weights, float *dnextLayer,
                                     float *dcurrentLayer, int dim_W, int dim_nextLayer)
{
    //int col =
    //int row =

    // TO DO  : SUM OF ERRORS
}

// Without BATCH LEARNING ( no sum to do, no average) : ONLINE
__global__ void __kernel_update_weights(float *weights, float *nextLayer, float *dcurrentLayer, float learning_rate)
{
    /*int col =
    int row =

            //Weight errors
            dwij = nextLayer[TO DO ]
    //Update weight
    weights[TO DO] = weights[TO DO] - learning_rate * dwij

    //(3) Synchronization before next layer
    __syncThreads();*/
}

// Without BATCH LEARNING ( no sum to do, no average) : ONLINE
__global__ void __kernel_update_biases(float *biases, float *nextLayer, float *dcurrentLayer, float learning_rate)
{
    /*
    int col =
    int row =

            //Weight errors
            dwij = nextLayer[TO DO]
    //Update weight
    biases[TO DO] = biases[TO DO] - learning_rate * dwij

    //(3) Synchronization before next layer
    __syncThreads();
    */
}

__global__ void __kernel_backward_propagation(float *errors)
{
    /*size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;*/

    // Update the parameters of the activation functions.
    //__kernel_layer_error(errors); //TODO
}


/**
 * Wrappers for call on host.
 */


void layer_cuda::backward_propagation(dim3 block_dims, dim3 thread_dims, float *errors)
{
    __kernel_backward_propagation<<<block_dims, thread_dims>>>(errors);
    cudaDeviceSynchronize();
}
