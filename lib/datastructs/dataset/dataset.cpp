//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "dataset.h"


dataset::dataset()
{
}

dataset::dataset(std::vector<element> &elements):
    _elements(elements)
{
}

void dataset::add(const matrix &features, const matrix &labels)
{
    add(element(features, labels));
}

void dataset::add(const element &elem)
{
    _elements.push_back(elem);
}

void dataset::remove(const element &elem)
{
    // TODO
    /**
    _elements.erase(std::remove(_elements.begin(), _elements.end(), elem), 
            _elements.end());
            */
}

void dataset::remove(const size_t i)
{
    //TODO
    /*
    _elements.erase(_elements.begin() + i);
    */
}

const element &dataset::get(const size_t i) const
{
    return _elements.at(i);
}

const std::vector<element> &dataset::get_elements() const
{
    return _elements;
}

size_t dataset::size() const
{
    return _elements.size();
}

matrix dataset::get_features() const
{
    matrix features(4, 4);

    // TODO return all the features concatenated.
    return features;
}

matrix dataset::get_labels() const
{
    matrix labels(4, 4);

    // TODO return all the labels concatenated.
    return labels;
}

std::pair<dataset, dataset> dataset::train_test_split(float train_size_ratio /*= 0.8f*/)
{
    if (train_size_ratio < 0 || 1 < train_size_ratio)
    {
        // Invalid.
        util::ERROR("dataset::train_test_split", "Invalid @training_size_ratio");
        util::ERROR_EXIT();
    }

    size_t size_ = size();

    if (size_ < 2)
    {
        // Invalid.
        util::ERROR("dataset::train_test_split", "Dataset is too small");
        util::ERROR_EXIT();
    }

    dataset train;
    dataset test;
    size_t train_size = size_ * train_size_ratio;
    size_t nb_selected = 0;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution = std::normal_distribution<float>(0.f, 1.f);

    for (size_t i = 0; nb_selected < train_size; i ++)
    {
        float probability_to_be_selected = (train_size - nb_selected) 
                / (size_ - i);
        float random = ((float) distribution(generator) 
                / (float) (RAND_MAX)) * (float) (size_ - i);

        if (random <= probability_to_be_selected)
        {
            train.add(get(i));
            nb_selected ++;
        }
        else
        {
            test.add(get(i));
        }
    }

    return std::pair<dataset, dataset>(train, test);
}

dataset dataset::get_random_batch(size_t batch_size)
{
    size_t size_ = size();

    if (batch_size < 0 || size_ < batch_size)
    {
        // Invalid.
        util::ERROR("dataset::get_random_batch", "Invalid @batch_size");
        util::ERROR_EXIT();
    }

    dataset batch;
    size_t nb_selected = 0;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution = std::normal_distribution<float>(0.f, 1.f);

    for (size_t i = 0; (i < size_) && (nb_selected < batch_size); i ++)
    {
        float probability_to_be_selected = (batch_size - nb_selected) 
                / (size_ - i);
        float random = ((float) distribution(generator) 
                / (float) (RAND_MAX)) * (float) (size_ - i);

        if (random <= probability_to_be_selected)
        {
            batch.add(get(i));
            nb_selected ++;
        }
    }

    return batch;
}

dataset dataset::load_mult()
{
    dataset data; 

    for (size_t i = 0; i < MULT_SIZE; i ++)
    {
        // TODO need to free pointer.
        auto features = new matrix(1, MULT_NB_FEATURES); 
        auto labels = new matrix({ 1 }, 1, MULT_NB_LABELS);

        for (size_t j = 0; j < MULT_NB_FEATURES; j ++)
        {
            features->get_host_data()[j] = (int) (std::rand() % MULT_MAX);
            labels->get_host_data()[0] *= (int) features->get_host_data()[j]; 
        }

        data.add(*features, *labels);
    }

    return data;
}

void dataset::print(const dataset &d)
{
    size_t i = 1;

    for (auto &e: d.get_elements())
    {
        std::cout << "*** n°" << i ++ << " ***" << std::endl;
        element::print(e);
    }
}
