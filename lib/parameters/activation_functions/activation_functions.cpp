//
// Created by Emilien Aufauvre on 29/11/2021.
//

#include "activation_functions.h"


float linear(float input, float a = 1.f, float b = 1.f)
{
    return a * input + b;
}

float binary_step(float input, float threshold = 0.f)
{
    return input < threshold ? 0.f : 1.f;
}

float sigmoid(float input)
{
    return 1.f / 1.f + exp(-input);
}

float relu(float input)
{
    return std::max(0.f, input);
}
