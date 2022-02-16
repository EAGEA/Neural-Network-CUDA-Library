//
// Created by Emilien Aufauvre on 07/01/2022.
//

#include "lib/functions/activation_functions/activation_functions.h"


using namespace cudaNN;


#define x 5
#define y 5


/**
 * Basic calls of activation functions on matrices.
 * Includes CUDA examples.
 */
int main(int argc, char *argv[])
{
    // ----------- //
    std::cout << "> original matrix" << std::endl;
    auto m1 = matrix(x, y, "original");
    for (size_t i = 0; i < x; i ++)
    {
        for (size_t j= 0; j < y; j ++)
        {
            m1[i * y + j] = -1.f + i * j;
        }
    }
    matrix::print(m1);
    // ----------- //
    std::cout << "> linear (identity)" << std::endl;
    matrix::print(activation_functions::LINEAR.compute({&m1}));
    matrix::print(activation_functions::LINEAR.compute_derivatives({&m1}));
    // ----------- //
    std::cout << "> binary step" << std::endl;
    matrix::print(activation_functions::BINARY_STEP.compute({&m1}));
    matrix::print(activation_functions::BINARY_STEP.compute_derivatives({&m1}));
    // ----------- //
    std::cout << "> sigmoid" << std::endl;
    matrix::print(activation_functions::SIGMOID.compute({&m1}));
    matrix::print(activation_functions::SIGMOID.compute_derivatives({&m1}));
    // ----------- //
    std::cout << "> relu" << std::endl;
    matrix::print(activation_functions::RELU.compute({&m1}));
    matrix::print(activation_functions::RELU.compute_derivatives({&m1}));
    // ----------- //
    std::cout << "> tanh" << std::endl;
    matrix::print(activation_functions::TANH.compute({&m1}));
    matrix::print(activation_functions::TANH.compute_derivatives({&m1}));
    // ----------- //
    std::cout << "> softmax" << std::endl;
    matrix::print(activation_functions::SOFTMAX.compute({&m1}));
    matrix::print(activation_functions::SOFTMAX.compute_derivatives({&m1}));
    // ----------- //
    return EXIT_SUCCESS;
}