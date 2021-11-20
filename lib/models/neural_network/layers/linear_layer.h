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
         * TODO CHECK IF INDEXES ARE TRUE
         * Parameters of the activation functions.
         * The parameters of the neuron at position (i, j) in the layer,
         * are at position (i, j) in the matrices,
         * where i < "_dimensions.first" and j < "_dimensions.second".
         */
        matrix _biases;
        matrix _weights;
};


#endif //CUDANN_LINEAR_LAYER_H