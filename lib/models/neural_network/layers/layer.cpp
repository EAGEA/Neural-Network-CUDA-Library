//
// Created by Emilien Aufauvre on 02/11/2021.
//


layer::layer(std::pair<size_t, sizet> dimensions)
{
    _dimensions = dimensions;
}

matrix layer::forward_propagation(matrix features)
{
    // TODO preparation
    _kernel_forward_propagation(features);
}

matrix layer::backward_propagation(matrix errors)
{
    // TODO preparation
    _kernel_backward_propagation(features);
    _update_biases(errors);
    _update_weights(errors);
}

const std::pair<size_t, size_t> get_dimensions() const
{
    return _dimensions;
}