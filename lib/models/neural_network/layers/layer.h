//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LAYER_H
#define CUDANN_LAYER_H


/**
 * Layer of neurons to be included in a neural network.
 */
class layer
{
    public:

        layer(std::pair<size_t, size_t> dimensions);
        virtual matrix forward_propagation(matrix features);
        virtual matrix back_propagation(matrix errors)

        const std::pair<size_t, size_t> get_dimensions();

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
         * The parameters of the neuron at position (i, j) in the layer,
         * are at position (i, j) in the matrices,
         * where i < "_dimensions.first" and j < "_dimensions.second".
         */
        const std::pair<size_t, size_t> _dimensions;
        matrix _biases;
        matrix _weights;
};


#endif //CUDANN_LAYER_H