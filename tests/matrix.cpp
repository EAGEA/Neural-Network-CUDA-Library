//
// Created by Emilien Aufauvre on 10/12/2021.
//

#include "lib/datastructs/matrix/matrix.h"


using namespace cudaNN;
using namespace cudaNN::matrix_operators;


#define x 5
#define y 5


/**
 * Examples of basic operations on matrices with basic operators
 * and constructors.
 * Includes CUDA tests.
 */
int main(int argc, char *argv[])
{
    // ----------- //
    std::cout << "> [] operator" << std::endl;
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
    std::cout << "> Copy constructor" << std::endl;
    auto m2 = m1;
    matrix::print(m2);
    // ----------- //
    std::cout << "> = operator" << std::endl;
    m2 = matrix(m1, "2");
    m2.set_id("2");
    matrix::print(m2);
    // ----------- //
    std::cout << "> == operator" << std::endl;
    std::cout << "m1 is equal to m2: "
              << (m1 == m2 ? "true" : "false")
              << std::endl;
    // ----------- //
    std::cout << "> += operator" << std::endl;
    m1 += m2;
    matrix::print(m1);
    // ----------- //
    std::cout << "> + & = operators" << std::endl;
    m1 = m1 + m2;
    matrix::print(m1);
    // ----------- //
    std::cout << "> *= operator" << std::endl;
    m2 *= m2;
    matrix::print(m2);
    // ----------- //
    std::cout << "> * & = operators" << std::endl;
    m1 = m1 * m2;
    matrix::print(m1);
    // ----------- //
    std::cout << "> != operator" << std::endl;
    std::cout << "m1 is not equal to m2: "
              << (m1 != m2 ? "true" : "false")
              << std::endl;
    // ----------- //
    std::cout << "> init list constructor" << std::endl;
    matrix::print(matrix({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3, "3"));
    // ----------- //

    return EXIT_SUCCESS;
}