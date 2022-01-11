//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "layer.h"

#include <utility>


using namespace cudaNN;


layer::layer(const size_t input_size, const size_t nb_neurons,
             function activation_function):
        _size(nb_neurons),
        _biases(1, nb_neurons, "layer::biases"),
        _weights(input_size, nb_neurons, "layer::weights"),
        _activation_function(std::move(activation_function))
{
    _init_biases();
    _init_weights();
}

void layer::_init_biases()
{
    for (int x = 0; x < _biases.get_dimensions().second; x ++)
    {
        _biases[x] = 0.f;
    }
}

void layer::_init_weights()
{
    std::random_device generator;
    std::normal_distribution<float> distribution = std::normal_distribution<float>(0.f, 1.f);

    for (int i = 0; i < _weights.get_length(); i ++)
    {
        _weights[i] = distribution(generator);
    }
}

matrix layer::feed_forward(matrix &inputs)
{
    if (inputs.get_dimensions().second != _weights.get_dimensions().first)
    {
        // Invalid.
        util::ERROR("layer::_feed_forward",
                    "Invalid @inputs size ("
                    + std::to_string(inputs.get_dimensions().second)
                    + " instead of "
                    + std::to_string(_weights.get_dimensions().first)
                    + ")");
        util::ERROR_EXIT();
    }

    // Compute the output of each neuron.
    auto sum = inputs * _weights + _biases;
    // Compute the result of the activation function derivative on the inputs (for backprop).
    _derivative = _activation_function.compute_derivative({ &sum });
    // Compute the result of the activation function on the inputs.
    return _activation_function.compute({ &sum });
}

void layer::backward_propagation(matrix &errors, const layer *previous)
{
    if (previous != nullptr)
    {
        // If not the output layer.
        errors = previous->get_weights().transpose() * errors;
    }

    // Do the vector (and not matrix) mult computation "errors * _derivative".
    auto cuda_dims = util::get_cuda_dims(errors.get_dimensions());
    layer_cuda::backward_propagation(cuda_dims.first, cuda_dims.second,
                                     errors, _derivative);
}

matrix layer::get_weights() const
{
    return _weights;
}

size_t layer::size() const
{
    return _size;
}