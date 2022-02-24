//
// Created by Hugo on 15/01/2022.
//

#include "lib/data_structures/matrix/matrix.h"

#include <fstream>


using namespace cudaNN;


#define MIN_POWER 0
#define MAX_POWER 10
#define NB_OPERATIONS 7


// To record execution time depending on if we are using GPU or not.
#if _USE_GPU
void wrapper(matrix (matrix::*f)(const matrix&), matrix &m1, matrix &m2, float &time_event)
{
    cudaEvent_t start_event, end_event;

    util::GPU_start_record(start_event, end_event);
    (m1.*f)(m2);
    util::GPU_end_record(&time_event, start_event, end_event);
}
void wrapper(matrix (matrix::*f)(float), matrix &m, float &time_event)
{
    cudaEvent_t start_event, end_event;

    util::GPU_start_record(start_event, end_event);
    (m.*f)(0.1f);
    util::GPU_end_record(&time_event, start_event, end_event);
}
void wrapper(matrix (matrix::*f)() const, matrix &m, float &time_event)
{
    cudaEvent_t start_event, end_event;

    util::GPU_start_record(start_event, end_event);
    (m.*f)();
    util::GPU_end_record(&time_event, start_event, end_event);
}
void wrapper(float (matrix::*f)() const, matrix &m, float &time_event)
{
    cudaEvent_t start_event, end_event;

    util::GPU_start_record(start_event, end_event);
    (m.*f)();
    util::GPU_end_record(&time_event, start_event, end_event);
}
#else
void wrapper(matrix (matrix::*f)(const matrix&), matrix &m1, matrix &m2, float &time_event)
{
    util::CPU_start_record(&time_event);
    (m1.*f)(m2);
    util::CPU_end_record(&time_event);
}
void wrapper(matrix (matrix::*f)(float), matrix &m, float &time_event)
{
    util::CPU_start_record(&time_event);
    (m.*f)(0.1f);
    util::CPU_end_record(&time_event);
}
void wrapper(matrix (matrix::*f)() const, matrix &m, float &time_event)
{
    util::CPU_start_record(&time_event);
    (m.*f)();
    util::CPU_end_record(&time_event);
}
void wrapper(float (matrix::*f)() const, matrix &m, float &time_event)
{
    util::CPU_start_record(&time_event);
    (m.*f)();
    util::CPU_end_record(&time_event);
}
#endif


/**
 * Compute the total execution time of the operations on matrices,
 * for different matrix sizes.
 * Execute the operations either on the device or host
 * depending on the -global.h/_USE_GPU- variable.
 * Output them in .csv files to be plotted.
 */
int main(int argc, char *argv[])
{
    std::ofstream csv[NB_OPERATIONS];

    for (size_t i = 0; i < NB_OPERATIONS; i ++)
    {
        csv[i].open(std::to_string(i) + "_matrix_"
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

        wrapper(&matrix::operator+, m1, m2, time_event[0]);
        wrapper(&matrix::operator-, m1, m2, time_event[0]);
        wrapper(&matrix::operator*, m1, m2, time_event[0]);
        wrapper(&matrix::operator*, m1, time_event[0]);
        wrapper(&matrix::hadamard_product, m1, m2, time_event[0]);
        wrapper(&matrix::sum, m1, time_event[0]);
        wrapper(&matrix::transpose, m1, time_event[0]);

        // Add to the files.
        for (size_t j = 0; j < NB_OPERATIONS; j ++)
        {
            csv[j] << std::to_string(i) + ";" + std::to_string(time_event[j]) + "\n";
        }

        std::cout << "Operations on 2^" << i << " × 2^" << i << " matrices over." << std::endl;
    }

    for (auto &item: csv)
    {
        item.close();
    }
}
