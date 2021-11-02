//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_DATASET_H
#define CUDANN_DATASET_H


class dataset
{
    public:

        dataset();
        dataset(std::vec<element> elements);

        void add(matrix features, matrix labels);
        void add(element elem);
        void remove(element elem);
        void remove(size_t i);

        std::vec<element> get_elements();
        const size_t size();

        /**
         * @param train_size represent the proportion of the training dataset
         * (between 0.f and 1.f).
         * @return a dataset for training, and another to test.
         */
        std::pair<dataset, dataset> train_test_split(float train_size = 0.8f);

        /**
         * @param batch_size
         * @return a random batch of size "batch_size".
         */
        dataset get_random_batch(size_t batch_size);

        /**
         * @return and load the MNIST dataset.
         */
        static dataset loadMNIST();

    private:

        std::vec<element> _elements;
};


#endif //CUDANN_DATASET_H