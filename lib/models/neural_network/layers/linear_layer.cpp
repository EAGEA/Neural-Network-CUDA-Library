//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "linear_layer.h"


linear_layer::linear_layer(const size_t input_size, const size_t nb_neurons,
                           const activation_function_t activation_function): 
    layer(nb_neurons), 
    _biases(1, nb_neurons), 
    _weights(input_size, nb_neurons),
    _activation_function(activation_function)
{
    _init_biases();
    _init_weights();
}

void linear_layer::_init_biases()
{
    for (int x = 0; x < _biases.get_dimensions().second; x ++)
    {
        _biases[x] = 0.f;
    }
}

void linear_layer::_init_weights()
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution = std::normal_distribution<float>(0.f, 1.f);

    for (int i = 0; i < _weights.get_dimensions().first; i ++)
    {
        for (int j = 0; j < _weights.get_dimensions().second; j ++)
        {
            _weights[i * _weights.get_dimensions().second + j] = distribution(generator);
        }
    }
}

matrix linear_layer::forward_propagation(const matrix &inputs)
{
    if (inputs.get_dimensions().second != _weights.get_dimensions().first)
    {
        // Invalid.
        util::ERROR("linear_layer::forward_propagation", "Invalid @inputs size");
        util::ERROR_EXIT();
    }

    // Compute the output of each neuron.
    return _activation_function(inputs * _weights + _biases);
}

matrix linear_layer::backward_propagation(const matrix &errors)
{
    // TODO
    auto new_errors = matrix(errors.get_dimensions());
    auto cuda_dims = util::get_cuda_dims(1, 1); // TODO choose dims

    __backward_propagation(cuda_dims.first, cuda_dims.second, new_errors.get_device_data());

    return new_errors;
}
