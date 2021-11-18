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

        matrix forward_propagation(matrix features);
        matrix backward_propagation(matrix errors);

        const std::pair<size_t, size_t> get_dimensions() const;

    private:

        virtual matrix _kernel_forward_propagation(matrix features);
        virtual matrix _kernel_backward_propagation(matrix errors);

        /**
         * Used during backpropagation, update the weights accordingly to the "errors".
         * @param errors
         */
        virtual void _update_biases(matrix errors);

        /**
         * Used during backpropagation, update the biases accordingly to the "errors".
         * @param errors
         */
        virtual void _update_weights(matrix errors);

        /**
         * Dimensions of the layer.
         */
        const std::pair<size_t, size_t> _dimensions;
};


#endif //CUDANN_LAYER_H