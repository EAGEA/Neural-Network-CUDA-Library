//
// Created by Hugo on 15/01/2022.
//

#include "lib/data_structures/matrix/matrix.h"
#include "lib/functions/activation_functions/activation_functions.h"
#include "lib/functions/loss_functions/loss_functions.h"

#include <fstream>


using namespace cudaNN;


#define STEP 32
#define MIN_SIZE STEP
#define MAX_SIZE 512
#define NB_FUNCTIONS (12 * 2)


// To record execution time depending on if we are using GPU or not.
#if _USE_GPU
void wrapper(const function& f, matrix &m1, matrix &m2, float &time_event_1, float &time_event_2)
{
    cudaEvent_t start_event, end_event;

    util::GPU_start_record(start_event, end_event);
    f.compute({ &m1, &m2 });
    util::GPU_end_record(&time_event_1, start_event, end_event);

    util::GPU_start_record(start_event, end_event);
    f.compute_derivatives({ &m1, &m2 });
    util::GPU_end_record(&time_event_2, start_event, end_event);
}
#else
void wrapper(const function &f, matrix &m1, matrix &m2, float &time_event_1, float &time_event_2)
{
    util::CPU_start_record(&time_event_1);
    f.compute({ &m1, &m2 });
    util::CPU_end_record(&time_event_1);

    util::CPU_start_record(&time_event_2);
    f.compute_derivatives({ &m1, &m2 });
    util::CPU_end_record(&time_event_2);
}
#endif


/**
 * Compute the total execution time of the functions (loss and activation),
 * for different matrix sizes.
 * Execute the operations either on the device or host
 * depending on the -global.h/_USE_GPU- variable.
 * Output them in .csv files to be plotted.
 */
int main(int argc, char *argv[])
{
    std::ofstream csv[NB_FUNCTIONS];

    for (size_t i = 0; i < NB_FUNCTIONS; i ++)
    {
        csv[i].open(std::to_string(i) + "_functions_"
                    + std::string(_USE_GPU ? "gpu" : "cpu") + ".csv");
        csv[i] << "Nb elements;" + std::string(_USE_GPU ? "GPU" : "CPU") + " Time\n";
    }

    float time_event[NB_FUNCTIONS];

    for (size_t i = MIN_SIZE; i <= MAX_SIZE; i += STEP)
    {
        auto m1 = matrix(i, i, "1");
        auto m2 = matrix(i, i, "2");

        // Activation functions.
        wrapper(activation_functions::LINEAR, m1, m2, time_event[0], time_event[1]);
        wrapper(activation_functions::BINARY_STEP, m1, m2, time_event[2], time_event[3]);
        wrapper(activation_functions::SIGMOID, m1, m2, time_event[4], time_event[5]);
        wrapper(activation_functions::RELU, m1, m2, time_event[6], time_event[7]);
        wrapper(activation_functions::TANH, m1, m2, time_event[8], time_event[9]);
        wrapper(activation_functions::SOFTMAX, m1, m2, time_event[10], time_event[11]);
        // Loss functions.
        wrapper(loss_functions::MEAN_SQUARED_ERROR, m1, m2, time_event[12], time_event[13]);
        wrapper(loss_functions::MEAN_ABSOLUTE_ERROR, m1, m2, time_event[14], time_event[15]);
        wrapper(loss_functions::MEAN_BIAS_ERROR, m1, m2, time_event[16], time_event[17]);
        wrapper(loss_functions::HINGE_LOSS, m1, m2, time_event[18], time_event[19]);
        wrapper(loss_functions::BINARY_CROSS_ENTROPY_LOSS, m1, m2, time_event[20], time_event[21]);
        wrapper(loss_functions::CROSS_ENTROPY_LOSS, m1, m2, time_event[22], time_event[23]);

        // Add to the files.
        for (size_t j = 0; j < NB_FUNCTIONS; j ++)
        {
            csv[j] << std::to_string(i) + ";" + std::to_string(time_event[j]) + "\n";
        }

        std::cout << "Functions on " << i << " × " << i << " matrices over." << std::endl;
    }

    for (auto &item: csv)
    {
        item.close();
    }
}
