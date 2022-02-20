//
// Created by Hugo on 19/02/2022.
//

#include "lib/data_structures/matrix/matrix.h"

//Weights dimensions of L1 (matrix)
#define weights_l1_x 8
#define weights_l1_y 4

//Errors dimensions of L1 (vector)
#define errors_l1_x 4
#define errors_l1_y 1

//Errors dimensions of L0 (vector)
#define errors_l0_x 8
#define errors_l0_y 1

//Inputs dimensions of L0 (vector)
#define inputs_x 8
#define inputs_y 1

//Learning rate
#define lr 0.1

using namespace cudaNN;

int main(int argc, char *argv[])
{
    //Weights init for the layer L1
    auto weights_l1 = matrix(weights_l1_x, weights_l1_y, "1");

    for(int i = 0 ; i < weights_l1_x ; i++)
    {
        for(int j = 0 ; j < weights_l1_y; j++)
        {
            weights_l1[i * weights_l1_y + j] = 0.5;
        }
    }

    //Errors init for the layer L1
    auto errors_l1 = matrix(errors_l1_x,errors_l1_y,"2");

    for(int i = 0 ; i < errors_l1_x ; i ++)
    {
        errors_l1[i] = 0.2;
    }

    //Errors init for the layer L0
    auto errors_l0 = matrix(errors_l0_x,errors_l0_y,"3");

    for(int i = 0 ; i < errors_l0_x ; i ++)
    {
        errors_l0[i] = 0.2;
    }

    //Inputs init (layer L0)
    auto inputs = matrix(inputs_x,inputs_y,"4");

    for(int i = 0 ; i < inputs_x ; i ++)
    {
        inputs[i] = 0.3;
    }
    //LAYER INITIALISATION (BEFORE)
    std::cout << "BEFORE : WEIGHTS \n";
    matrix::print(weights_l1);
    std::cout << "BEFORE : ERRORS L1 \n";
    matrix::print(errors_l1);
    std::cout << "BEFORE : ERRORS L0 \n";
    matrix::print(errors_l0);
    std::cout << "BEFORE : INPUTS L1 \n";
    matrix::print(inputs);

    //ERROR PROPAGATION
    auto errors = weights_l1 * errors_l1;
    matrix::print(errors);
    errors_l0 = errors.hadamard_product(inputs.transpose());
    std::cout << "AFTER : errors_l0 \n";
    matrix::print(errors_l0);

    //WEIGHT UPDATE SIMULATION (AFTER)
    std::cout << "AFTER : delta_weights \n";
    auto delta_weights = (inputs * errors_l1.transpose()) * lr;
    matrix::print(delta_weights);
    weights_l1 = weights_l1 - delta_weights;
    std::cout << "AFTER : weights \n";
    matrix::print(weights_l1);
}