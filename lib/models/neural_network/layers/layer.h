//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LAYER_H
#define CUDANN_LAYER_H

#include "lib/models/neural_network/layers/layer.h"
#include "lib/parameters/activation_functions/activation_function.h"
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
                  activation_function activation_function);

            matrix forward_propagation(matrix &inputs) const;
            matrix backward_propagation(const matrix &errors);

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
             * The parameters of the neuron n°i in the layer are
             * at the column n°i in the matrices.
             * "_weights" has the same number of row as the number of features,
             * and the same number of columns as the number of neurons.
             * "_biases" has the same number of columns as the number of neurons,
             * and has only one row.
             */
            matrix _biases;
            matrix _weights;

            /**
             * Parameters for the backpropagation
             * "_old_weights" is the copy of "_weights" to keep in memory for
             * error propagation (after weights update)
             * "_old_biases" is the copy of "_biases" to keep in memory for
             * error propagation (after biases update)
             * "_previous_layer" corresponds to the previous layer in the
             * backpropagation direction.
             * "_dpreviousLayer" corresponds to the error of the previous
             * layer.
             * "_dcurrentLayer" corresponds to the error of the current
             * layer.
             */

            matrix _old_weights;
            matrix _old_biases;
            matrix _previous_layer;
            matrix _d_previous_layer;
            matrix _d_current_layer;


            const activation_function _activation_function;

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
        void backward_propagation(dim3 block_dims, dim3 thread_dims, float *errors);
    }
}


#endif //CUDANN_LAYER_H
