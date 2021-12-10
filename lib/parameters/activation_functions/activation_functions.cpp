//
// Created by Emilien Aufauvre on 29/11/2021.
//

#include "activation_functions.h"


float activation_functions::linear(float input)
{
    return input;
}

float activation_functions::binary_step(float input)
{
    return input < 0.f ? 0.f : 1.f;
}

float activation_functions::sigmoid(float input)
{
    return 1.f / 1.f + exp(-input);
}

float activation_functions::relu(float input)
{
    return fmax(0.f, input);
}
