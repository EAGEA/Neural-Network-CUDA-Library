//
// Created by Emilien Aufauvre on 20/11/2021.
//

#ifndef CUDANN_LOSS_FUNCTIONS_H
#define CUDANN_LOSS_FUNCTIONS_H

#include "lib/datastructs/matrix/matrix.h"
#include "lib/util/util.h"


namespace cudaNN
{
    /**
     * Compute and return the error between two matrices;
     * "predictions" and "labels" (the ground truth).
     */

    typedef matrix (*loss_function_t)(const matrix &, const matrix &);


    namespace loss_functions
    {
        // For regression.
        matrix mean_square_error(const matrix &predictions, const matrix &labels);
        matrix mean_absolute_error(const matrix &predictions, const matrix &labels);
        matrix mean_bias_error(const matrix &predictions, const matrix &labels);
        // For classification.
        matrix svm_loss(const matrix &predictions, const matrix &labels);
        matrix cross_entropy_loss(const matrix &predictions, const matrix &labels);
    }


    /**
     * CUDA function wrappers for call on host.
     */
    namespace loss_functions_cuda
    {
        void mean_square_error(dim3 block_dims, dim3 thread_dims,
                               matrix &results,
                               matrix &predictions,
                               matrix &errors);
        void mean_absolute_error(dim3 block_dims, dim3 thread_dims,
                                 matrix &results,
                                 matrix &predictions,
                                 matrix &errors);
        void mean_bias_error(dim3 block_dims, dim3 thread_dims,
                             matrix &results,
                             matrix &predictions,
                             matrix &errors);
        void svm_loss(dim3 block_dims, dim3 thread_dims,
                      matrix &results,
                      matrix &predictions,
                      matrix &errors);
        void cross_entropy_loss(dim3 block_dims, dim3 thread_dims,
                                matrix &results,
                                matrix &predictions,
                                matrix &errors);
    }
}


#endif //CUDANN_LOSS_FUNCTIONS_H
