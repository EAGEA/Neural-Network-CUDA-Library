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

        layer(size_t nb_neurons);

        virtual matrix forward_propagation(matrix features);
        virtual matrix backward_propagation(matrix errors);

        const size_t size() const;

    private:

        /**
         * Dimension of the layer.
         */
        const size_t _size;
};


#endif //CUDANN_LAYER_H