//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "layer.h"


using namespace cudaNN;


layer::layer(const size_t input_size, const size_t nb_neurons,
             initializations init,
             const function &activation_function):
        _size(nb_neurons),
        _biases(1, nb_neurons, "layer::biases"),
        _weights(input_size, nb_neurons, "layer::weights"),
        _activation_function(activation_function),
        _first_entry(true)
{
    _init_biases();
    _init_weights(init);
}

void layer::_init_biases()
{
    for (int x = 0; x < _biases.get_dimensions().second; x ++)
    {
        _biases[x] = 0.f;
    }
}

void layer::_init_weights(initializations init)
{
    std::random_device generator;
    std::normal_distribution<float> distribution;

    switch (init)
    {
        case initializations::XAVIER:
            distribution = std::normal_distribution<float>(
                    0.f, sqrtf(1.f / (float) _weights.get_dimensions().first));
            break;
        case initializations::HE:
            distribution = std::normal_distribution<float>(
                    0.f, sqrtf(2.f / (float) _weights.get_dimensions().first));
            break;
    }

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
    _derivatives = _activation_function.compute_derivatives({ &sum });
    // Compute the result of the activation function on the inputs.
    return _activation_function.compute({ &sum });
}

void layer::backward_propagation(matrix &errors, layer *next)
{
    if (next != nullptr)
    {
        // If not the output layer.
        errors = next->_weights * errors.transpose();
    }

    errors = errors.hadamard_product(_derivatives.transpose());
    if (_first_entry)
    {
        // The first entry of the batch (i.e. first computed errors).
        _errors = errors;
        _first_entry = false;
    }
    else
    {
        _errors += errors;
    }

    if(next == nullptr)
    {
        _errors = _errors.transpose();
    }
}

void layer::gradient_descent(size_t batch_size, float learning_rate)
{
    // Update weights and biases.
    _weights -= (_errors * _inputs).transpose() * (learning_rate / (float) batch_size);
    _biases -= _errors.transpose() * (learning_rate / (float) batch_size);
    // Reset for next backpropagation.
    _first_entry = true;
}

void layer::print_neurons()
{
    matrix::print(_inputs);
}

void layer::print_weights()
{
    matrix::print(_weights);
}

void layer::print_biases()
{
    matrix::print(_biases);
}

void layer::print_errors()
{
    matrix::print(_errors);
}


size_t layer::size() const
{
    return _size;
}