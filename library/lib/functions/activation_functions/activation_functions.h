//
// Created by Emilien Aufauvre on 10/01/2022.
//

#ifndef CUDANN_ACTIVATION_FUNCTIONS_H
#define CUDANN_ACTIVATION_FUNCTIONS_H

#include "lib/data_structures/matrix/matrix.h"
#include "lib/functions/function.h"


namespace cudaNN
{
    /**
     * Cuda functions to be executed on device.
     */
    namespace activation_functions_parallel
    {
        void linear(std::vector<matrix *> m);
        void linear_derivative(std::vector<matrix *> m);
        void binary_step(std::vector<matrix *> m);
        void binary_step_derivative(std::vector<matrix *> m);
        void sigmoid(std::vector<matrix *> m);
        void sigmoid_derivative(std::vector<matrix *> m);
        void relu(std::vector<matrix *> m);
        void relu_derivative(std::vector<matrix *> m);
        void tanh(std::vector<matrix *> m);
        void tanh_derivative(std::vector<matrix *> m);
        void softmax(std::vector<matrix *> m);
    }


    /**
     * C++ functions to be executed on host.
     */
    namespace activation_functions_sequential
    {
        void linear(std::vector<matrix *> m);
        void linear_derivative(std::vector<matrix *> m);
        void binary_step(std::vector<matrix *> m);
        void binary_step_derivative(std::vector<matrix *> m);
        void sigmoid(std::vector<matrix *> m);
        void sigmoid_derivative(std::vector<matrix *> m);
        void relu(std::vector<matrix *> m);
        void relu_derivative(std::vector<matrix *> m);
        void tanh(std::vector<matrix *> m);
        void tanh_derivative(std::vector<matrix *> m);
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
#if _USE_GPU
        using namespace activation_functions_parallel;
#else
        using namespace activation_functions_sequential;
#endif

        const auto LINEAR = function("linear",
                                     linear,
                                     linear_derivative);

        const auto BINARY_STEP = function("binary",
                                          binary_step,
                                          binary_step_derivative);

        const auto SIGMOID = function("sigmoid",
                                      sigmoid,
                                      sigmoid_derivative);

        const auto RELU = function("relu",
                                   relu,
                                   relu_derivative);

        const auto TANH = function("tanh",
                                   tanh,
                                   tanh_derivative);
        const auto SOFTMAX = function("softmax",
                                   softmax,
                                   tanh_derivative);
    }
}


#endif //CUDANN_ACTIVATION_FUNCTIONS_H