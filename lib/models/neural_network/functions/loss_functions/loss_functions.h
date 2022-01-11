//
// Created by Emilien Aufauvre on 20/11/2021.
//

#ifndef CUDANN_LOSS_FUNCTIONS_H
#define CUDANN_LOSS_FUNCTIONS_H

#include "lib/data_structures/matrix/matrix.h"
#include "lib/models/neural_network/functions/function.h"


namespace cudaNN
{
    /**
     * CUDA function wrappers for call on host.
     * Execute the named function on device.
     */
    namespace loss_functions_cuda
    {
        void mean_squared_error(dim3 block_dims, dim3 thread_dims,
                                std::vector<matrix *> m);
        void mean_squared_error_derivative(dim3 block_dims, dim3 thread_dims,
                                           std::vector<matrix *> m);
        void mean_absolute_error(dim3 block_dims, dim3 thread_dims,
                                 std::vector<matrix *> m);
        void mean_absolute_error_derivative(dim3 block_dims, dim3 thread_dims,
                                            std::vector<matrix *> m);
        void mean_bias_error(dim3 block_dims, dim3 thread_dims,
                             std::vector<matrix *> m);
        void mean_bias_error_derivative(dim3 block_dims, dim3 thread_dims,
                                        std::vector<matrix *> m);
        void hinge_loss(dim3 block_dims, dim3 thread_dims,
                        std::vector<matrix *> m);
        void hinge_loss_derivative(dim3 block_dims, dim3 thread_dims,
                                   std::vector<matrix *> m);
        void binary_cross_entropy_loss(dim3 block_dims, dim3 thread_dims,
                                       std::vector<matrix *> m);
        void binary_cross_entropy_loss_derivative(dim3 block_dims, dim3 thread_dims,
                                                  std::vector<matrix *> m);
    }


    /**
     * Compute and return the error between two matrices;
     * "predictions" and "labels" (the ground truths).
     */
    namespace loss_functions
    {
        using namespace loss_functions_cuda;

        /**
         * For regression.
         * Predictions which are far away from actual values are
         * penalized heavily in comparison to less deviated predictions.
         * Direction of the error is not considered.
         */
        const auto MEAN_SQUARED_ERROR = function("mean_squared_error",
                                                 mean_squared_error,
                                                 mean_squared_error_derivative);

        /**
         * For regression.
         * Predictions which are far away from actual values are
         * penalized heavily in comparison to less deviated predictions.
         * Direction of the error is not considered.
         */
        const auto MEAN_ABSOLUTE_ERROR = function("mean_absolute_error",
                                                  mean_absolute_error,
                                                  mean_absolute_error_derivative);

        /**
         * For regression.
         * Caution as positive and negative errors could cancel each other out.
         * Direction of the error is considered.
         */
        const auto MEAN_BIAS_ERROR = function("mean_bias_error",
                                              mean_bias_error,
                                              mean_bias_error_derivative);

        /**
         * For "maximum-margin" classification.
         * Also named "SVM Loss" (used by SVM).
         * Measure the performance of a classification model whose output
         * is a probability value between 0 and 1.
         * Increase as the predicted probability diverges from the actual label.
         */
        const auto HINGE_LOSS = function("hinge_loss",
                                         hinge_loss,
                                         hinge_loss_derivative);

        /**
         * For binary classification.
         * Measure the performance of a classification model whose output
         * is a probability value between 0 and 1.
         * Increase as the predicted probability diverges from the actual label.
         */
        const auto BINARY_CROSS_ENTROPY_LOSS = function("binary_cross_entropy_loss",
                                                        binary_cross_entropy_loss,
                                                        binary_cross_entropy_loss_derivative);
    }
}


#endif //CUDANN_LOSS_FUNCTIONS_H