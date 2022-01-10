//
// Created by Emilien Aufauvre on 20/11/2021.
//

#ifndef CUDANN_LOSS_FUNCTIONS_H
#define CUDANN_LOSS_FUNCTIONS_H

#include "lib/data_structures/matrix/matrix.h"
#include "lib/util/util.h"


namespace cudaNN
{
    /**
     * Compute and return the error between two matrices;
     * "predictions" and "labels" (the ground truth).
     */

    typedef float (*loss_function_t)(const matrix &, const matrix &);


    namespace loss_functions
    {
        /**
         * For regression.
         * Predictions which are far away from actual values are
         * penalized heavily in comparison to less deviated predictions.
         * Direction of the error is not considered.
         * @param predictions - output of the model.
         * @param labels - ground truth.
         * @return - the computed error.
         */
        float mean_squared_error(const matrix &predictions, const matrix &labels);

        /**
         * For regression.
         * Predictions which are far away from actual values are
         * penalized heavily in comparison to less deviated predictions.
         * Direction of the error is not considered.
         * @param predictions - output of the model.
         * @param labels - ground truth.
         * @return - the computed error.
         */
        float mean_absolute_error(const matrix &predictions, const matrix &labels);

        /**
         * For regression.
         * Caution as positive and negative errors could cancel each other out.
         * Direction of the error is considered.
         * @param predictions - output of the model.
         * @param labels - ground truth.
         * @return - the computed error.
         */
        float mean_bias_error(const matrix &predictions, const matrix &labels);

        /**
         * For "maximum-margin" classification.
         * Also named "SVM Loss" (used by SVM).
         * Measure the performance of a classification model whose output
         * is a probability value between 0 and 1.
         * Increase as the predicted probability diverges from the actual label.
         * @param predictions - output of the model.
         * @param labels - ground truth.
         * @return - the computed error.
         */
        float hinge_loss(const matrix &predictions, const matrix &labels);

        /**
         * For binary classification.
         * Measure the performance of a classification model whose output
         * is a probability value between 0 and 1.
         * Increase as the predicted probability diverges from the actual label.
         * @param predictions - output of the model.
         * @param labels - ground truth.
         * @return - the computed error.
         */
        float binary_cross_entropy_loss(const matrix &predictions, const matrix &labels);
    }


    /**
     * CUDA function wrappers for call on host.
     */
    namespace loss_functions_cuda
    {
        void mean_squared_error(dim3 block_dims, dim3 thread_dims,
                               const matrix &errors,
                               const matrix &predictions, const matrix &labels);
        void mean_absolute_error(dim3 block_dims, dim3 thread_dims,
                                 const matrix &errors,
                                 const matrix &predictions, const matrix &labels);
        void mean_bias_error(dim3 block_dims, dim3 thread_dims,
                             const matrix &errors,
                             const matrix &predictions, const matrix &labels);
        void hinge_loss(dim3 block_dims, dim3 thread_dims,
                      const matrix &errors,
                      const matrix &predictions, const matrix &labels);
        void binary_cross_entropy_loss(dim3 block_dims, dim3 thread_dims,
                                       const matrix &errors,
                                       const matrix &predictions, const matrix &labels);
    }
}


#endif //CUDANN_LOSS_FUNCTIONS_H
