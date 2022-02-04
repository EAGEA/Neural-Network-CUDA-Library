//
// Created by Hugo on 15/01/2022.
//

#include "lib/data_structures/matrix/matrix.h"

#include <fstream>


using namespace cudaNN;


#define MIN_POWER 0
#define MAX_POWER 11
#define NB_OPERATIONS 7


/**
 * Compute the total time of the operations on matrices.
 */
int main(int argc, char *argv[])
{
    std::ofstream csv[NB_OPERATIONS];

    for (size_t i = 0; i < NB_OPERATIONS; i ++)
    {
        csv[i].open("outputs/" + std::to_string(i) + "_"
                    + std::string(_USE_GPU ? "gpu" : "cpu") + ".csv");
        csv[i] << "Nb elements;" + std::string(_USE_GPU ? "GPU" : "CPU") + " Time\n";
    }

    float time_event[NB_OPERATIONS];
    cudaEvent_t start_event, end_event;

    for (size_t i = MIN_POWER; i <= MAX_POWER; i ++)
    {
        auto size = (size_t) std::pow(2.f, i);
        auto m1 = matrix(size, size, "1");
        auto m2 = matrix(size, size, "2");

        // Add.
        util::start_record(start_event, end_event);
        m1 + m2;
        util::end_record(&time_event[0], start_event, end_event);
        // Sub.
        util::start_record(start_event, end_event);
        m1 - m2;
        util::end_record(&time_event[1], start_event, end_event);
        // Multiply.
        util::start_record(start_event, end_event);
        m1 * m2;
        util::end_record(&time_event[2], start_event, end_event);
        // Multiply float.
        util::start_record(start_event, end_event);
        m1 * 1.f;
        util::end_record(&time_event[3], start_event, end_event);
        // Hadamard.
        util::start_record(start_event, end_event);
        m1.hadamard_product(m2);
        util::end_record(&time_event[4], start_event, end_event);
        // Sum.
        util::start_record(start_event, end_event);
        m1.sum();
        util::end_record(&time_event[5], start_event, end_event);
        // Transposes.
        util::start_record(start_event, end_event);
        m1.transpose();
        util::end_record(&time_event[6], start_event, end_event);

        // Add to the files.
        for (size_t j = 0; j < NB_OPERATIONS; j ++)
        {
            csv[j] << std::to_string(2 * i) + ";" + std::to_string(time_event[j]) + "\n";
        }

        std::cout << "Operations for 2^" << i << " × 2^" << i << " matrices over." << std::endl;
    }

    for (auto &item: csv)
    {
        item.close();
    }
}
