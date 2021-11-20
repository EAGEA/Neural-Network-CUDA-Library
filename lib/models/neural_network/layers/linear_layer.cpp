//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "linear_layer.h"


linear_layer::linear_layer(std::pair<size_t, size_t> dimensions)
: layer(dimensions)
{
    _biases = new matrix(dimensions);
    _weights = new matrix(dimensions);

    _init_biases();
    _init_weights();
}

matrix forward_propagation(matrix features)
{
    // TODO matrix preparation
    matrix outputs;

    __linear_forward_propagation();

    return outputs;
}

matrix backward_propagation(matrix errors)
{
    // TODO matrix preparation
    matrix new_errors;

    __linear_backward_propagation<<DIM_GRID, DIM_BLOCK>>();

    return new_errors;
}

void linear_layer::_init_biases()
{
    for (int x = 0; x < _dimensions.first; x ++)
    {
        for (int y = 0; y < _dimensions.second; y ++)
        {
            _biases[y * _dimensions.first + x] = 0.f;
        }
    }
}

void linear_layer::_init_weights()
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.f, 1.f);

    for (int x = 0; x < _dimensions.first; x ++)
    {
        for (int y = 0; y < _dimensions.second; y ++)
        {
            _weights[y * _dimensions.first + x] = distribution(generator);
        }
    }
}


size_t DIM_GRID = 6;
size_t DIM_BLOCK = 128;


// TODO try to name function without "linear", having multiple function with the same name across
// the layer files.
__global__ void __linear_forward_propagation(matrix features)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
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