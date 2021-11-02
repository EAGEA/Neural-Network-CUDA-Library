//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_MODEL_H
#define CUDANN_MODEL_H


/**
 * Model to be trained such that it identifies patterns,
 * and predict them on data.
 */
class model
{
    public:

        /**
         * Train the model on the given training dataset ("features", "labels").
         * @param data features and labels to work on.
         * @param epochs the number of iterations of the training process.
         * @param batch_size the size of the samples during the training.
         */
        virtual void fit(dataset data, epochs = 1, batch_size = 1);

        /**
         * @param features
         * @return the predictions of the model, on the given "features".
         */
        virtual matrix predict(matrix features);
};


#endif //CUDANN_MODEL_H