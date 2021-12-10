//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LINEAR_LAYER_H
#define CUDANN_LINEAR_LAYER_H

#include "lib/models/neural_network/layers/layer.h"
#include "lib/parameters/activation_functions/activation_functions.h"
#include "lib/util/util.h"
#include "/usr/local/cuda/include/vector_types.h"

#include <random>


/**
 * Layer of neurons using a linear activation function.
 */
class linear_layer: public layer
{
    public:

        /**
         * @param nb_neurons the total number of neurons in this layer.
         * @param nb_features the number of features of each entry of the dataset.
         * @param activation_function the function that compute the output of a neuron. 
         */
        linear_layer(const size_t nb_neurons, const size_t nb_features,
                     activation_function_t activation_function);

        matrix forward_propagation(matrix features) override;
        matrix backward_propagation(matrix errors) override;

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

        const activation_function_t _activation_function;
};

/**
 * CUDA function wrappers.
 */

void __execute_activation_functions(dim3 block_dims, dim3 thread_dims,
                                    activation_function_t activation_function,
                                    float *inputs, float *outputs,
                                    size_t nb_neurons);
void __backward_propagation(dim3 block_dims, dim3 thread_dims, float *errors);



#endif //CUDANN_LINEAR_LAYER_H
