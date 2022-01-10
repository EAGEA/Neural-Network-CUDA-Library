//
// Created by Emilien Aufauvre on 10/01/2022.
//

#ifndef CUDANN_ACTIVATION_FUNCTION_H
#define CUDANN_ACTIVATION_FUNCTION_H

#include "lib/data_structures/matrix/matrix.h"


namespace cudaNN
{
    /**
     * CUDA function wrappers for call on host.
     * Execute the named function on device.
     */
    namespace activation_functions_cuda
    {
        typedef void (*activation_function_t)(dim3 block_dims, dim3 thread_dims,
                                              const matrix &results, const matrix &inputs);

        void linear(dim3 block_dims, dim3 thread_dims,
                    const matrix &results, const matrix &inputs);
        void linear_derivative(dim3 block_dims, dim3 thread_dims,
                               const matrix &results, const matrix &inputs);
        void binary_step(dim3 block_dims, dim3 thread_dims,
                         const matrix &results, const matrix &inputs);
        void binary_step_derivative(dim3 block_dims, dim3 thread_dims,
                                    const matrix &results, const matrix &inputs);
        void sigmoid(dim3 block_dims, dim3 thread_dims,
                     const matrix &results, const matrix &inputs);
        void sigmoid_derivative(dim3 block_dims, dim3 thread_dims,
                                const matrix &results, const matrix &inputs);
        void relu(dim3 block_dims, dim3 thread_dims,
                  const matrix &results, const matrix &inputs);
        void relu_derivative(dim3 block_dims, dim3 thread_dims,
                             const matrix &results, const matrix &inputs);
    }


    /**
     * Compute and return the outputs of the activation function for each
     * nodes in a layer.
     * Each cell of the inputted matrix contains the input of a node.
     * Inputs(i) corresponds to addition between the weighted sum of the previous
     * layer outputs, and the bias of the neuron "i".
     * (i.e. (x1 * wi1) + ... + (x2 * wi2) + bi).
     */
    class activation_function
    {
        public:

            activation_function(std::string id,
                                activation_functions_cuda::activation_function_t function,
                                activation_functions_cuda::activation_function_t function_derivative);
            /**
             * @param input - the matrix on to be used.
             * @return - the result of the activation function "_function" on "input".
             */
            matrix compute(const matrix &inputs) const;

            /**
             * @param input - the matrix on to be used.
             * @return - the result of the derivative of the activation function
             * "_function_derivative" on "input".
             */
            matrix compute_derivative(const matrix &inputs) const;

        private:

            const std::string _id;
            const activation_functions_cuda::activation_function_t _function;
            const activation_functions_cuda::activation_function_t _function_derivative;
    };


    /**
     * Constants to be used.
     */
    namespace activation_functions
    {
        using namespace activation_functions_cuda;

        const activation_function LINEAR = activation_function("linear",
                                                               linear,
                                                               linear_derivative);
        const activation_function BINARY_STEP = activation_function("binary_step",
                                                                    binary_step,
                                                                    binary_step_derivative);
        const activation_function SIGMOID = activation_function("sigmoid",
                                                                sigmoid,
                                                                sigmoid_derivative);
        const activation_function RELU = activation_function("relu",
                                                             relu,
                                                             relu_derivative);
    }
}


#endif //CUDANN_ACTIVATION_FUNCTION_H