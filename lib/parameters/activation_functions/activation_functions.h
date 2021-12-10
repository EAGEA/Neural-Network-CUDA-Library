//
// Created by Emilien Aufauvre on 29/11/2021.
//

#ifndef CUDANN_ACTIVATION_FUNCTIONS_H
#define CUDANN_ACTIVATION_FUNCTIONS_H


/**
 * Compute the output of a node.
 * For a neural network, the input is the weighted
 * sum of the previous layer outputs, with the bias
 * corresponding to the current neuron
 * (i.e. (x1 * w1) + ... + (x2 * w2) + b).
 */

typedef float (*activation_function_t)(float);

namespace activation_functions
{
    float linear(float input); 
    float binary_step(float input);
    float sigmoid(float input);
    float relu(float input);
}


#endif //CUDANN_ACTIVATION_FUNCTIONS_H
