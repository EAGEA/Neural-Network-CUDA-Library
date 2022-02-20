//
// Created by Hugo on 19/02/2022.
//

#include "lib/data_structures/matrix/matrix.h"

//Weights dimensions of L1 (matrix)
#define weights_x 8
#define weights_y 4

//Errors dimensions of L1 (vector)
#define errors_x 4
#define errors_y 1

//Inputs dimensions of L0 (vector)
#define inputs_x 8
#define inputs_y 1

//Learning rate
#define lr 0.1

using namespace cudaNN;

int main(int argc, char *argv[])
{
    //Weights init for the layer L1
    auto weights = matrix(weights_x, weights_y, "1");

    for(int i = 0 ; i < weights_x ; i++)
    {
        for(int j = 0 ; j < weights_y; j++)
        {
            weights[i * weights_y + j] = 0.5;
        }
    }

    //Errors init for the layer L1
    auto errors = matrix(errors_x,errors_y,"2");

    for(int i = 0 ; i < errors_x ; i ++)
    {
        errors[i] = 0.1;
    }

    //Inputs init (layer L0)
    auto inputs = matrix(inputs_x,inputs_y,"3");

    for(int i = 0 ; i < inputs_x ; i ++)
    {
        inputs[i] = 0.2;
    }
    //Print of the layer BEFORE
    std::cout << "BEFORE : weights \n";
    matrix::print(weights);
    std::cout << "BEFORE : errors \n";
    matrix::print(errors);
    std::cout << "BEFORE : inputs \n";
    matrix::print(inputs);

    //Print the matrices when update weights AFTER
    std::cout << "AFTER : delta_weights \n";
    auto delta_weights = (inputs * errors.transpose()) * lr;
    matrix::print(delta_weights);
    weights = weights - delta_weights;
    std::cout << "AFTER : weights \n";
    matrix::print(weights);
}