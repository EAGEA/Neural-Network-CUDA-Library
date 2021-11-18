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