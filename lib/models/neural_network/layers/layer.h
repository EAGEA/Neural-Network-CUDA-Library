//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_LAYER_H
#define CUDANN_LAYER_H

#include "lib/datastructs/matrix/matrix.h" 

#include <cstddef>


/**
 * Layer of neurons to be included in a neural network.
 */
class layer
{
    public:

        explicit layer(const size_t nb_neurons);

        virtual matrix forward_propagation(matrix features) = 0;
        virtual matrix backward_propagation(matrix errors) = 0;

        const size_t size() const;

    protected:

        /**
         * Dimension of the layer.
         */
        const size_t _size;
};


#endif //CUDANN_LAYER_H
