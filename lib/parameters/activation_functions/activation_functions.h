//
// Created by Emilien Aufauvre on 29/11/2021.
//

#ifndef CUDANN_ACTIVATION_FUNCTIONS_H
#define CUDANN_ACTIVATION_FUNCTIONS_H

#include "lib/datastructs/matrix/matrix.h"

#include <cmath>


namespace cudaNN
{
    /**
     * Compute and return the outputs of the activation function
     * for each nodes in a layer.
     * Each cell of the inputted matrix contains the
     * input for a node.
     * For a neural network, the inputs are the weighted
     * sum of the previous layer outputs, with the bias
     * corresponding to the current neuron
     * (i.e. (x1 * w1) + ... + (x2 * w2) + b).
     */

    typedef matrix (*activation_function_t)(const matrix &);


    namespace activation_functions
    {
        matrix linear(const matrix &inputs);
        matrix binary_step(const matrix &inputs);
        matrix sigmoid(const matrix &inputs);
        matrix relu(const matrix &inputs);
    }


    /**
     * CUDA function wrappers for call on host.
     */
    namespace activation_functions_cuda
    {
        void linear(dim3 block_dims, dim3 thread_dims,
                    const matrix &results, const matrix &inputs);
        void binary_step(dim3 block_dims, dim3 thread_dims,
                         const matrix &results, const matrix &inputs);
        void sigmoid(dim3 block_dims, dim3 thread_dims,
                     const matrix &results, const matrix &inputs);
        void relu(dim3 block_dims, dim3 thread_dims,
                  const matrix &results, const matrix &inputs);
    }
}


#endif //CUDANN_ACTIVATION_FUNCTIONS_H