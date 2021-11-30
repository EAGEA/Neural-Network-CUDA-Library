//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LINEAR_LAYER_H
#define CUDANN_LINEAR_LAYER_H


/**
 * Layer of neurons using a linear activation function.
 */
class linear_layer: public layer
{
    public:

        /**
         * @param nb_neurons the total number of neurons in this layer.
         * @param nb_features the number of features of each entry of the dataset.
         */
        linear_layer(size_t nb_neurons, size_t nb_features);

        virtual matrix forward_propagation(matrix features) override;
        virtual matrix backward_propagation(matrix errors) override;

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

        activation_function _activation_function;
};


#endif //CUDANN_LINEAR_LAYER_H