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
     */
    namespace activation_functions_cuda
    {
        typedef void (*activation_function_t)(dim3 block_dims, dim3 thread_dims,
                                              const matrix &results, const matrix &inputs);

        void LINEAR(dim3 block_dims, dim3 thread_dims,
                    const matrix &results, const matrix &inputs);
        void LINEAR_DERIVATIVE(dim3 block_dims, dim3 thread_dims,
                               const matrix &results, const matrix &inputs);
        void BINARY_STEP(dim3 block_dims, dim3 thread_dims,
                         const matrix &results, const matrix &inputs);
        void BINARY_STEP_DERIVATIVE(dim3 block_dims, dim3 thread_dims,
                                    const matrix &results, const matrix &inputs);
        void SIGMOID(dim3 block_dims, dim3 thread_dims,
                     const matrix &results, const matrix &inputs);
        void SIGMOID_DERIVATIVE(dim3 block_dims, dim3 thread_dims,
                                const matrix &results, const matrix &inputs);
        void RELU(dim3 block_dims, dim3 thread_dims,
                  const matrix &results, const matrix &inputs);
        void RELU_DERIVATIVE(dim3 block_dims, dim3 thread_dims,
                             const matrix &results, const matrix &inputs);
    }


    /**
     * Compute and return the outputs of the activation function for each nodes in a layer.
     * Each cell of the inputted matrix contains the
     * input for a node.
     * For a neural network, the inputs are the weighted
     * sum of the previous layer outputs, with the bias
     * corresponding to the current neuron
     * (i.e. (x1 * w1) + ... + (x2 * w2) + b).
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
        const activation_function LINEAR = activation_function("linear",
                                                               activation_functions_cuda::LINEAR,
                                                               activation_functions_cuda::LINEAR_DERIVATIVE);
        /*
        matrix binary_step(const matrix &inputs);
        matrix sigmoid(const matrix &inputs);
        matrix relu(const matrix &inputs);
         */
    }
}


#endif //CUDANN_ACTIVATION_FUNCTION_H