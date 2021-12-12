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
     * Each cell of the inputed matrix contains the 
     * input for a node.
     * For a neural network, the inputs are the weighted
     * sum of the previous layer outputs, with the bias
     * corresponding to the current neuron
     * (i.e. (x1 * w1) + ... + (x2 * w2) + b).
     */

    typedef matrix (*activation_function_t)(matrix);


    namespace activation_functions
    {
        matrix linear(matrix inputs);
        matrix binary_step(matrix inputs);
        matrix sigmoid(matrix inputs);
        matrix relu(matrix inputs);
    }


    /**
     * CUDA function wrappers for call on host.
     */
    namespace activation_functions_cuda
    {
        void linear(dim3 block_dims, dim3 thread_dims,
                    float *results, float *inputs,
                    size_t nb_rows, size_t nb_cols);
        void binary_step(dim3 block_dims, dim3 thread_dims,
                         float *results, float *inputs,
                         size_t nb_rows, size_t nb_cols);
        void sigmoid(dim3 block_dims, dim3 thread_dims,
                     float *results, float *inputs,
                     size_t nb_rows, size_t nb_cols);
        void relu(dim3 block_dims, dim3 thread_dims,
                  float *results, float *inputs,
                  size_t nb_rows, size_t nb_cols);
    }
}


#endif //CUDANN_ACTIVATION_FUNCTIONS_H
