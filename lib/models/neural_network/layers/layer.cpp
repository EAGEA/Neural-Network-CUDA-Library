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
    // Save the inputs from previous layer.
    _inputs = inputs;
    // Compute the output of each neuron.
    auto sum = _inputs * _weights + _biases;
    // Compute the result of the activation function derivative on the inputs (for back propagation).
    _derivatives = _activation_function.compute_derivatives({&sum});
    // Compute the result of the activation function on the inputs.
    return _activation_function.compute({ &sum });
}

void layer::backward_propagation(matrix &errors, layer *next, float learning_rate)
{
    if (next != nullptr)
    {
        // If not the output layer.
        errors = next->_old_weights * errors;
    }

    errors = errors.hadamard_product(_derivatives.transpose());

    // Do the gradient descent.
    // - update weights.
    _old_weights = _weights;
    _weights -= (errors * _inputs).transpose() * learning_rate;
    // - update biases.
    _biases -= errors.transpose() * learning_rate;
}

size_t layer::size() const
{
    return _size;
}