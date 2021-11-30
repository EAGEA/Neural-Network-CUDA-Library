//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "linear_layer.h"


linear_layer::linear_layer(size_t nb_neurons, size_t nb_features,
                           activation_function_t activation_function)
: layer(nb_neurons)
{
    _biases = new matrix(1, nb_neurons);
    _weights = new matrix(nb_features, nb_neurons);
    _activation_function = activation_function;

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

    matrix inputs, outputs;
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(_dimensions.first,
                                                          _dimensions.second);

    // Compute entries of activation functions.
    inputs = features * _weights + _biases;
    // Compute outputs of the same function.
    __kernel_execute_activation_functions<<cuda_dims.first, cuda_dims.second>>(
                    _activation_function,
                    inputs.get_device_data(),
                    outputs.get_device_data()) ;

    return outputs;
}

matrix linear_layer::backward_propagation(matrix errors)
{
    // TODO
    matrix new_errors;

    __linear_backward_propagation<<DIM_GRID, DIM_BLOCK>>();

    return new_errors;
}


/**
 * CUDA
 */


__global__ void __kernel_execute_activation_functions(activation_function_t activation_function,
                                                      float *inputs, float *outputs,
)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    outputs[row * nb_cols + col] = activation_function(inputs[row * nb_cols + col]);
}

__global__ void __linear_backward_propagation(matrix errors)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Update the parameters of the activation functions.
    __update_activation_functions()<<DIM_GRID, DIM_BLOCK>>(errors);
}

__device__ void __update_activation_functions(matrix errors)
{
    // Update the weights.

    // Update the biases.

}