//
// Created by Emilien Aufauvre on 20/11/2021.
//

#ifndef CUDANN_LOSS_FUNCTIONS_H
#define CUDANN_LOSS_FUNCTIONS_H


/**
 * Compute the error between two matrices;
 * "predictions" and "labels" (the ground truth).
 */

typedef matrix (*loss_function_t)(matrix, matrix);

namespace loss_functions
{
    // For regression.
    matrix mean_square_error(matrix predictions, matrix labels);
    matrix mean_absolute_error(matrix predictions, matrix labels);
    matrix mean_bias_error(matrix predictions, matrix labels);
    // For classification.
    matrix svm_loss(matrix predictions, matrix labels);
    matrix cross_entropy_loss(matrix predictions, matrix labels);
}


#endif //CUDANN_LOSS_FUNCTIONS_H