//
// Created by Hugo on 15/01/2022.
//

#include "lib/data_structures/matrix/matrix.h"

#include <fstream>


using namespace cudaNN;


#define MIN_POWER 0
#define MAX_POWER 10
#define NB_OPERATIONS 7


/**
 * Compute the total time of the operations on matrices.
 * Execute the operations either on the device or host
 * depending on the -global.h/_USE_GPU- variable.
 * Output them in .csv files to be plotted.
 */
int main(int argc, char *argv[])
{
    std::ofstream csv[NB_OPERATIONS];

    for (size_t i = 0; i < NB_OPERATIONS; i ++)
    {
        csv[i].open(std::to_string(i) + "_"
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
        util::CPU_start_record(&time_event[0]);
        m1 + m2;
        util::CPU_end_record(&time_event[0]);
        // Sub.
        util::CPU_start_record(&time_event[1]);
        m1 - m2;
        util::CPU_end_record(&time_event[1]);
        // Multiply.
        if (_USE_GPU)
        {
            util::GPU_start_record(start_event, end_event);
            m1 * m2;
            util::GPU_end_record(&time_event[2], start_event, end_event);
        }
        else
        {
            util::CPU_start_record(&time_event[2]);
            m1 * m2;
            util::CPU_end_record(&time_event[2]);
        }
        // Multiply float.
        util::CPU_start_record(&time_event[3]);
        m1 * 1.f;
        util::CPU_end_record(&time_event[3]);
        // Hadamard.
        util::CPU_start_record(&time_event[4]);
        m1.hadamard_product(m2);
        util::CPU_end_record(&time_event[4]);
        // Sum.
        if (_USE_GPU)
        {
            util::GPU_start_record(start_event, end_event);
            m1.sum();
            util::GPU_end_record(&time_event[5], start_event, end_event);
        }
        else
        {
            util::CPU_start_record(&time_event[5]);
            m1.sum();
            util::CPU_end_record(&time_event[5]);
        }
        // Transposes.
        if (_USE_GPU)
        {
            util::GPU_start_record(start_event, end_event);
            m1.transpose();
            util::GPU_end_record(&time_event[6], start_event, end_event);
        }
        else
        {
            util::CPU_start_record(&time_event[6]);
            m1.transpose();
            util::CPU_end_record(&time_event[6]);
        }

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
