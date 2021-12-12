//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "loss_functions.h"


/**
 * Kernel functions.
 */


__global__ void __kernel_mean_square_error(float *results,
                                           float *predictions,
                                           float *errors)
{
    // TODO
}

__global__ void __kernel_mean_absolute_error(float *results,
                                             float *predictions,
                                             float *errors)
{
    // TODO
}

__global__ void __kernel_mean_bias_error(float *results,
                                         float *predictions,
                                         float *errors)
{
    // TODO
}

__global__ void __kernel_svm_loss(float *results,
                                  float *predictions,
                                  float *errors)
{
    // TODO
}

__global__ void __kernel_cross_entropy_loss(float *results,
                                            float *predictions,
                                            float *errors)
{
    // TODO
}


/**
 * Wrappers for call on host.
 */


void __loss_functions::__mean_square_error(dim3 block_dims, dim3 thread_dims,
                                           float *results,
                                           float *predictions,
                                           float *errors)
{
    __kernel_mean_square_error<<<block_dims, thread_dims>>>(
            results,
            predictions,
            errors);
}

void __loss_functions::__mean_absolute_error(dim3 block_dims, dim3 thread_dims,
                                             float *results,
                                             float *predictions,
                                             float *errors)
{
    __kernel_mean_absolute_error<<<block_dims, thread_dims>>>(
            results,
            predictions,
            errors);
}

void __loss_functions::__mean_bias_error(dim3 block_dims, dim3 thread_dims,
                                         float *results,
                                         float *predictions,
                                         float *errors)
{
    __kernel_mean_bias_error<<<block_dims, thread_dims>>>(
            results,
            predictions,
            errors);
}

void __loss_functions::__svm_loss(dim3 block_dims, dim3 thread_dims,
                                  float *results,
                                  float *predictions,
                                  float *errors)
{
    __kernel_svm_loss<<<block_dims, thread_dims>>>(
            results,
            predictions,
            errors);
}

void __loss_functions::__cross_entropy_loss(dim3 block_dims, dim3 thread_dims,
                                            float *results,
                                            float *predictions,
                                            float *errors)
{
    __kernel_cross_entropy_loss<<<block_dims, thread_dims>>>(
            results,
            predictions,
            errors);
}
