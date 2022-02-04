//
// Created by Hugo on 15/01/2022.
//

#include "lib/data_structures/matrix/matrix.h"


using namespace cudaNN;


#define x 256
#define y 256


/**
 * Compute the total time of the operations on matrices.
 */
int main(int argc, char *argv[])
{
    auto m1 = matrix(x, y, "1");
    auto m2 = matrix( x, y, "2");
    for (size_t i = 0; i < x; i ++)
    {
        for (size_t j= 0; j < y; j ++)
        {
            m1[i * y + j] = 3;
            m2[i * y + j] = 3;
        }
    }

    float time_event;
    cudaEvent_t start_event, end_event;

    util::start_record(start_event, end_event);
    m1 * m2;
    util::end_record(&time_event, start_event, end_event);
    std::cout << "Multiplication: " << time_event << "ms" << std::endl;

    util::start_record(start_event, end_event);
    m1 + m2;
    util::end_record(&time_event, start_event, end_event);
    std::cout << "Addition: " << time_event << "ms" << std::endl;

    util::start_record(start_event, end_event);
    m1 - m2;
    util::end_record(&time_event, start_event, end_event);
    std::cout << "Subtraction: " << time_event << "ms" << std::endl;

    util::start_record(start_event, end_event);
    m1.hadamard_product(m2);
    util::end_record(&time_event, start_event, end_event);
    std::cout << "Hadamard product: " << time_event << "ms" << std::endl;
}
