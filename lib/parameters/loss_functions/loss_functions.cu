//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "loss_functions.h"


/**
 * Kernel functions.
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


/**
 * Wrappers.
 */


void loss_functions::__mean_square_error(dim3 block_dims, dim3 thread_dims,
                         float *result,
                         float *predictions,
                         float *errors)
{
    __kernel_mean_square_error<<<block_dims, thread_dims>>>(
            result, 
            predictions, 
            errors);
}

void loss_functions::__mean_absolute_error(dim3 block_dims, dim3 thread_dims,
                           float *result,
                           float *predictions,
                           float *errors)
{
    __kernel_mean_absolute_error<<<block_dims, thread_dims>>>(
            result, 
            predictions, 
            errors);
}

void loss_functions::__mean_bias_error(dim3 block_dims, dim3 thread_dims,
                       float *result,
                       float *predictions,
                       float *errors)
{
    __kernel_mean_bias_error<<<block_dims, thread_dims>>>(
            result, 
            predictions, 
            errors);
}

void loss_functions::__svm_loss(dim3 block_dims, dim3 thread_dims,
                float *result,
                float *predictions,
                float *errors)
{
    __kernel_svm_loss<<<block_dims, thread_dims>>>(
            result, 
            predictions, 
            errors);
}

void loss_functions::__cross_entropy_loss(dim3 block_dims, dim3 thread_dims,
                          float *result,
                          float *predictions,
                          float *errors)
{
    __kernel_cross_entropy_loss<<<block_dims, thread_dims>>>(
            result, 
            predictions, 
            errors);
}
