//
// Created by Emilien Aufauvre on 20/11/2021.
//

#include "loss_functions.h"


matrix mean_square_error(matrix predictions, matrix labels)
{
    matrix error;
    // TODO allocate matrix
    __kernel_mean_square_error(error.get_device_data(),
                               predictions.get_device_data(),
                               labels.get_device_data());

    return error;
}

matrix mean_absolute_error(matrix predictions, matrix labels)
{
    matrix error;
    // TODO allocate matrix
    __kernel_mean_absolute_error(error.get_device_data(),
                               predictions.get_device_data(),
                               labels.get_device_data());

    return;
}

matrix mean_bias_error(matrix predictions, matrix labels)
{
    matrix error;
    // TODO allocate matrix
    __kernel_mean_bias_error(error.get_device_data(),
                               predictions.get_device_data(),
                               labels.get_device_data());

    return;
}

matrix svm_loss(matrix predictions, matrix labels)
{
    matrix error;
    // TODO allocate matrix
    __kernel_svm_loss(error.get_device_data(),
                             predictions.get_device_data(),
                             labels.get_device_data());

    return;
}

matrix cross_entropy_loss(matrix predictions, matrix labels)
{
    matrix error;
    // TODO allocate matrix
    __kernel_cross_entropy_loss(error.get_device_data(),
                      predictions.get_device_data(),
                      labels.get_device_data());

    return;
}


/**
* CUDA.
*/


__global__ void __kernel_mean_square_error(float *result,
                                           float *predictions,
                                           float *errors)
{
    // TODO
}

__global__ void __kernel_mean_absolute_error(float *result,
                                             float *predictions,
                                             float *errors)
{
    // TODO
}

__global__ void __kernel_mean_bias_error(float *result,
                                         float *predictions,
                                         float *errors)
{
    // TODO
}

__global__ void __kernel_svm_loss(float *result,
                                  float *predictions,
                                  float *errors)
{
    // TODO
}

__global__ void __kernel_cross_entropy_loss(float *result,
                                            float *predictions,
                                            float *errors)
{
    // TODO
}