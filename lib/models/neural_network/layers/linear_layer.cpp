//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "linear_layer.h"


linear_layer::linear_layer(size_t nb_neurons, size_t nb_features)
: layer(nb_neurons)
{
    _biases = new matrix(1, nb_neurons);
    _weights = new matrix(nb_features, nb_neurons);

    _init_biases();
    _init_weights();
}

void linear_layer::_init_biases()
{
    for (int x = 0; x < _dimensions.second; x ++)
    {
        _biases[x] = 0.f;
    }
}

void linear_layer::_init_weights()
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.f, 1.f);

    for (int i = 0; i < _dimensions.first; i ++)
    {
        for (int j = 0; j < _dimensions.second; j ++)
        {
            _weights[i * _dimensions.second + j] = distribution(generator);
        }
    }
}

matrix linear_layer::forward_propagation(matrix features)
{
    if (features.get_dimensions().first != _weights.get_dimensions().first)
    {
        // Invalid.
        util::print_error("linear_layer::forward_propagation", "Invalid @features size");
        util::exit_error();
    }

    // Compute entries of activation functions.
    matrix inputs = features * _weights + _biases;
    // Compute output of the same functions.
    matrix outputs = ;// TODO compute output of activation function;

    return outputs;
}

matrix linear_layer::backward_propagation(matrix errors)
{
    // TODO
    matrix new_errors;

    __linear_backward_propagation<<DIM_GRID, DIM_BLOCK>>();

    return new_errors;
}

__global__ void __linear_backward_propagation(matrix errors)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Update the parameters of the activation functions.
    __update_activation_functions()<<DIM_GRID, DIM_BLOCK>>(errors);
}

__device__ void __update_activation_functions(matrix errors)
{
    // Update the weights.

    // Update the biases.

}