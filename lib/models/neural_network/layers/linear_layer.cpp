//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "linear_layer.h"


linear_layer::linear_layer(const size_t nb_neurons, const size_t nb_features,
                           const activation_function_t activation_function): 
    layer(nb_neurons), 
    _biases(1, nb_neurons), 
    _weights(nb_features, nb_neurons),
    _activation_function(activation_function)
{
    _init_biases();
    _init_weights();
}

void linear_layer::_init_biases()
{
    for (int x = 0; x < _biases.get_dimensions().second; x ++)
    {
        // TODO device or host ??
        //_biases[x] = 0.f;
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
            // TODO device or host ??
            //_weights[i * _weights.get_dimensions().second + j] = distribution(generator);
        }
    }
}

matrix linear_layer::forward_propagation(matrix features)
{
    if (features.get_dimensions().second != _weights.get_dimensions().second)
    {
        // Invalid.
        util::ERROR("linear_layer::forward_propagation", "Invalid @features size");
        util::ERROR_EXIT();
    }

    matrix inputs = matrix(features.get_dimensions()); //TODO check dims
    matrix outputs = matrix(1, _size);
    // TODO allocate matrices
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(1, _size);

    // Compute entries of activation functions.
    inputs = features * _weights + _biases;
    // Compute outputs of the same functions.
    __execute_activation_functions(
                    cuda_dims.first, cuda_dims.second,
                    _activation_function,
                    inputs.get_device_data(),
                    outputs.get_device_data(),
                    _size) ;

    return outputs;
}

matrix linear_layer::backward_propagation(matrix errors)
{
    // TODO
    matrix new_errors = matrix(errors.get_dimensions());
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(1, 1); // TODO choose dims

    __backward_propagation(cuda_dims.first, cuda_dims.second, new_errors.get_device_data());

    return new_errors;
}
