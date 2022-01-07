//
// Created by Emilien Aufauvre on 07/01/2022.
//

#include "lib/parameters/activation_functions/activation_functions.h"


using namespace cudaNN;
using namespace cudaNN::matrix_operators;


#define x 5
#define y 5


/**
 * Basic calls of activation functions on matrices.
 * Includes CUDA tests.
 */
int main(int argc, char *argv[])
{
    // ----------- //
    std::cout << "> original matrix" << std::endl;
    auto m1 = matrix(x, y, "1");
    for (size_t i = 0; i < x; i ++)
    {
        for (size_t j= 0; j < y; j ++)
        {
            m1[i * y + j] = i * j;
        }
    }
    matrix::print(m1);
    // ----------- //
    std::cout << "> linear (identity)" << std::endl;
    matrix::print(activation_functions::linear(m1));
    // ----------- //
    std::cout << "> sigmoid" << std::endl;
    matrix::print(activation_functions::sigmoid(m1));
    // ----------- //
    std::cout << "> binary step" << std::endl;
    matrix::print(activation_functions::binary_step(m1));
    // ----------- //
    std::cout << "> relu" << std::endl;
    matrix::print(activation_functions::relu(m1));
    // ----------- //

    return EXIT_SUCCESS;
}