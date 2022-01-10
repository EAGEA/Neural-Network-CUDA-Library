//
// Created by Emilien Aufauvre on 07/01/2022.
//

#include "lib/parameters/loss_functions/loss_functions.h"


using namespace cudaNN;


#define x 5
#define y 5


/**
 * Basic calls of loss functions on matrices.
 * Includes CUDA tests.
 */
int main(int argc, char *argv[])
{
    // ----------- //
    std::cout << "> prediction matrix" << std::endl;
    auto m1 = matrix(x, y, "predictions");
    for (size_t i = 0; i < x; i ++)
    {
        for (size_t j= 0; j < y; j ++)
        {
            m1[i * y + j] = 0.01f * (1.f + i * j);
        }
    }
    matrix::print(m1);
    // ----------- //
    std::cout << "> labels matrix" << std::endl;
    auto m2 = matrix(x, y, "labels");
    for (size_t i = 0; i < x; i ++)
    {
        for (size_t j= 0; j < y; j ++)
        {
            m2[i * y + j] = -0.02f * (1.f + i * j);
        }
    }
    matrix::print(m2);
    // ----------- //
    std::cout << "> mean squared error (MSE)" << std::endl;
    std::cout << loss_functions::mean_squared_error(m1, m2) << std::endl;
    // ----------- //
    std::cout << "> mean absolute error (MAE)" << std::endl;
    std::cout << loss_functions::mean_absolute_error(m1, m2) << std::endl;
    // ----------- //
    std::cout << "> mean bias error (MBE)" << std::endl;
    std::cout << loss_functions::mean_bias_error(m1, m2) << std::endl;
    // ----------- //
    std::cout << "> hinge loss" << std::endl;
    std::cout << loss_functions::hinge_loss(m1, m2) << std::endl;
    // ----------- //
    std::cout << "> binary cross entropy loss (LCE)" << std::endl;
    std::cout << loss_functions::binary_cross_entropy_loss(m1, m2) << std::endl;
    // ----------- //

    return EXIT_SUCCESS;
}