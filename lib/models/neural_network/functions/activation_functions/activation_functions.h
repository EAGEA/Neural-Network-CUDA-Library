//
// Created by Emilien Aufauvre on 10/01/2022.
//

#ifndef CUDANN_ACTIVATION_FUNCTIONS_H
#define CUDANN_ACTIVATION_FUNCTIONS_H

#include "lib/data_structures/matrix/matrix.h"
#include "lib/models/neural_network/functions/function.h"


namespace cudaNN
{
    /**
     * CUDA function wrappers for call on host.
     * Execute the named function on device.
     */
    namespace activation_functions_cuda
    {
        void linear(dim3 block_dims, dim3 thread_dims,
                    std::vector<matrix *> m);
        void linear_derivative(dim3 block_dims, dim3 thread_dims,
                               std::vector<matrix *> m);
        void binary_step(dim3 block_dims, dim3 thread_dims,
                         std::vector<matrix *> m);
        void binary_step_derivative(dim3 block_dims, dim3 thread_dims,
                                    std::vector<matrix *> m);
        void sigmoid(dim3 block_dims, dim3 thread_dims,
                     std::vector<matrix *> m);
        void sigmoid_derivative(dim3 block_dims, dim3 thread_dims,
                                std::vector<matrix *> m);
        void relu(dim3 block_dims, dim3 thread_dims,
                  std::vector<matrix *> m);
        void relu_derivative(dim3 block_dims, dim3 thread_dims,
                             std::vector<matrix *> m);
    }


    /**
     * Compute and return the outputs of the activation function for each
     * nodes in a layer.
     * Each cell of the inputted matrix contains the input of a node.
     * Inputs(i) corresponds to addition between the weighted sum of the previous
     * layer outputs, and the bias of the neuron "i".
     * (i.e. (x1 * wi1) + ... + (x2 * wi2) + bi).
     */
    namespace activation_functions
    {
        using namespace activation_functions_cuda;

        const auto LINEAR = function("linear",
                                     linear,
                                     linear_derivative);
        const auto BINARY_STEP = function("binary_step",
                                          binary_step,
                                          binary_step_derivative);
        const auto SIGMOID = function("sigmoid",
                                      sigmoid,
                                      sigmoid_derivative);
        const auto RELU = function("relu",
                                   relu,
                                   relu_derivative);
    }
}


#endif //CUDANN_ACTIVATION_FUNCTIONS_H