//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LAYER_H
#define CUDANN_LAYER_H

#include "lib/models/neural_network/layers/layer.h"
#include "lib/models/neural_network/functions/activation_functions/activation_functions.h"
#include "lib/util/util.h"

#include <random>


namespace cudaNN
{
    /**
     * Layer of neurons, to be included in a neural network. 
     */
    class layer
    {
        public:

            /**
             * @param input_size - the size (number of columns) of the input.
             * @param nb_neurons - the total number of neurons in this layer.
             * @param activation_function - the function that compute the output of a neuron.
             */
            layer(const size_t input_size, const size_t nb_neurons,
                  function activation_function);

            matrix feed_forward(matrix &inputs);
            void backward_propagation(matrix &errors, const layer *previous);

            matrix get_weights() const;
            size_t size() const;

        private:

            /**
             * Initialize the "_biases" of the layer at 0
             * (most appropriate method in literature).
             */
            void _init_biases();

            /**
             * Initialize the "_weights" of the layer using the normal distribution
             * (most appropriate method in literature).
             */
            void _init_weights();

            /**
             * Parameters of the activation functions.
             * The functions of the neuron n°i in the layer are
             * at the column n°i in the matrices.
             * "_weights" has the same number of row as the number of features,
             * and the same number of columns as the number of neurons.
             * "_biases" has the same number of columns as the number of neurons,
             * and has only one row.
             */
            matrix _biases;
            matrix _weights;

            /**
             * To store the results of the derivative of the activation function
             * on the current inputs. Used during backpropagation.
             */
            matrix _derivative;

            const function _activation_function;

            /**
             * Dimension of the layer (number of neurons).
             */
            const size_t _size;
    };


    /**
     * CUDA function wrappers for call on host.
     */
    namespace layer_cuda
    {
        void backward_propagation(dim3 block_dims, dim3 thread_dims,
                                  const matrix &v1, const matrix &v2);
    }
}


#endif //CUDANN_LAYER_H
