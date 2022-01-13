//
// Created by Emilien Aufauvre on 07/01/2022.
//

#include "lib/functions/loss_functions/loss_functions.h"


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
    matrix::print(loss_functions::MEAN_SQUARED_ERROR.compute({ &m1, &m2 }));
    matrix::print(loss_functions::MEAN_SQUARED_ERROR.compute_derivatives({&m1, &m2}));
    // ----------- //
    std::cout << "> mean absolute error (MAE)" << std::endl;
    matrix::print(loss_functions::MEAN_ABSOLUTE_ERROR.compute({ &m1, &m2 }));
    matrix::print(loss_functions::MEAN_ABSOLUTE_ERROR.compute_derivatives({&m1, &m2}));
    // ----------- //
    std::cout << "> mean bias error (MBE)" << std::endl;
    matrix::print(loss_functions::MEAN_BIAS_ERROR.compute({ &m1, &m2 }));
    matrix::print(loss_functions::MEAN_BIAS_ERROR.compute_derivatives({&m1, &m2}));
    // ----------- //
    std::cout << "> hinge loss" << std::endl;
    matrix::print(loss_functions::HINGE_LOSS.compute({ &m1, &m2 }));
    matrix::print(loss_functions::HINGE_LOSS.compute_derivatives({&m1, &m2}));
    // ----------- //
    std::cout << "> binary cross entropy loss (LCE)" << std::endl;
    matrix::print(loss_functions::BINARY_CROSS_ENTROPY_LOSS.compute({ &m1, &m2 }));
    matrix::print(loss_functions::BINARY_CROSS_ENTROPY_LOSS.compute_derivatives({&m1, &m2}));
    // ----------- //

    return EXIT_SUCCESS;
}