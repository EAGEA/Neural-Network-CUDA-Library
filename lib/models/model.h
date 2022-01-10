//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_MODEL_H
#define CUDANN_MODEL_H

#include "lib/data_structures/dataset/dataset.h"
#include "lib/parameters/loss_functions/loss_functions.h"


namespace cudaNN
{
    /**
     * Model to be trained such that it identifies patterns,
     * and predict them on data.
     */
    class model
    {
        public:

            /**
             * Train the model on the given training dataset ("features", "labels").
             * @param data - features and labels to work on.
             * @param loss_function - function that may be used to calculate cost
             * during backpropagation of neural networks for example.
             * @param epochs - the number of iterations of the training process.
             * @param batch_size - the size of the samples during the training.
             * @param learning_rate - the learning rate for training
             */
            virtual void fit(dataset &data,
                             loss_function_t loss_function,
                             size_t epochs = 1,
                             size_t batch_size = 1,
                             float learning_rate = 0.01) = 0;

            /**
             * @param features - the features of a dataset entry.
             * @return - the predictions of the model, on the given "features".
             */
            virtual matrix predict(const matrix &features) const = 0;

            /**
             * @param test - a dataset for testing prediction abilities.
             * @return - the predictions of the model, on the given dataset.
             */
            virtual std::vector<matrix> predict(dataset &test) const = 0;
    };
}


#endif //CUDANN_MODEL_H
