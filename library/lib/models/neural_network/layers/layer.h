//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LAYER_H
#define CUDANN_LAYER_H

#include "lib/models/neural_network/layers/layer.h"
#include "lib/functions/activation_functions/activation_functions.h"
#include "lib/util/util.h"

#include <random>


namespace cudaNN
{
    /**
     * Weights type of initialization.
     * @XAVIER for tanh activation.
     * @HE for RELU activation.
     */
    enum initializations
    {
        XAVIER,
        HE
    };


    /**
     * Layer of neurons, to be included in a neural network. 
     */
    class layer
    {
        public:

            /**
             * @param input_size - the size (number of columns) of the input.
             * @param nb_neurons - the total number of neurons in this layer.
             * @param init - the type of weight initialization.
             * @param activation_function - the function that compute the output of a neuron.
             */
            layer(const size_t input_size, const size_t nb_neurons,
                  initializations init,
                  const function &activation_function);

            matrix feed_forward(matrix &inputs);
            void backward_propagation(matrix &errors, layer *next);
            void gradient_descent(size_t batch_size, float learning_rate);

            /**
             * Printing functions of the layer
             */
            void print_neurons();
            void print_weights();
            void print_biases();
            void print_errors();

            size_t size() const;

        private:

            /**
             * Initialize the "_biases" of the layer at 0
             * (most appropriate method in literature).
             */
            void _init_biases();

            /**
             * Initialize the "_weights" of the layer using the
             * specified method.
             * @param init - the type of initialization.
             */
            void _init_weights(initializations init);

            /**
             * Dimension of the layer (number of neurons).
             */
            const size_t _size;

            /**
             * Compute the output of the neuron.
             */
            const function &_activation_function;

            /**
             * Parameters to compute the input of the activation function.
             * The functions of the neuron n°i in the layer are
             * at the column n°i in the matrices.
             * @_weights - has the same number of row as the number of features,
             * and the same number of columns as the number of neurons.
             * @_biases - has the same number of columns as the number of neurons,
             * and has only one row.
             */
            matrix _biases;
            matrix _weights;

            /**
             * Parameters of the backpropagation and gradient descent.
             * @_derivatives - to store the results of the derivative of the activation
             * function on the current inputs.
             * @_inputs - to store the current inputs (the outputs from previous layer).
             * @_errors - the sum of the errors for the batch. Computed during the
             * backpropagation.
             * @_first_entry - true if it is the errors on the first entry of the batch that
             * are currently processed during the backpropagation process.
             */
            matrix _derivatives;
            matrix _inputs;
            matrix _errors;
            bool _first_entry;
    };
}


#endif //CUDANN_LAYER_H
