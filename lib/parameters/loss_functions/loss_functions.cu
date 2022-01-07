//
// Created by Emilien Aufauvre on 09/12/2021.
//

#include "loss_functions.h"


using namespace cudaNN;


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

void helper(dim3 block_dims, dim3 thread_dims,
            matrix &results,
            matrix &predictions,
            matrix &errors,
            void (kernel)(float *result, float *inputs, size_t nb_rows, size_t nb_cols))
{

}


/**
 * Wrappers for call on host.
 */


void loss_functions_cuda::mean_square_error(dim3 block_dims, dim3 thread_dims,
                                            matrix &results,
                                            matrix &predictions,
                                            matrix &errors)
{
}

void loss_functions_cuda::mean_absolute_error(dim3 block_dims, dim3 thread_dims,
                                              matrix &results,
                                              matrix &predictions,
                                              matrix &errors)
{
}

void loss_functions_cuda::mean_bias_error(dim3 block_dims, dim3 thread_dims,
                                          matrix &results,
                                          matrix &predictions,
                                          matrix &errors)
{
}

void loss_functions_cuda::svm_loss(dim3 block_dims, dim3 thread_dims,
                                   matrix &results,
                                   matrix &predictions,
                                   matrix &errors)
{
}

void loss_functions_cuda::cross_entropy_loss(dim3 block_dims, dim3 thread_dims,
                                             matrix &results,
                                             matrix &predictions,
                                             matrix &errors)
{
}