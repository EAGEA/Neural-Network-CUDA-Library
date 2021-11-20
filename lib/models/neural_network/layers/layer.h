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
        virtual matrix backward_propagation(matrix errors);

        const std::pair<size_t, size_t> get_dimensions() const;

    private:

        /**
         * Dimensions of the layer.
         */
        const std::pair<size_t, size_t> _dimensions;
};


#endif //CUDANN_LAYER_H