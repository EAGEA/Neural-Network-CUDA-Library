//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "dataset.h"


using namespace cudaNN;


dataset::dataset()
{
}

dataset::dataset(std::vector<element> &elements):
        _elements(elements)
{
}

void dataset::add(const matrix &features, const matrix &labels)
{
    // TODO pointer ?
    _elements.push_back(element(features, labels));
}

void dataset::add(element &elem)
{
    _elements.push_back(elem);
}

void dataset::set(const size_t index, element &elem)
{
    /*
    std::copy(_elements.begin() + index * sizeof(element), 
            _elements.begin() + (index + 1) * sizeof(element), 
            elem);
            */
    _elements[index] = elem;
}

void dataset::remove(element &elem)
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

element &dataset::get(const size_t i)
{
    return _elements[i];
}

std::vector<element> &dataset::get_elements()
{
    return _elements;
}

size_t dataset::size() const
{
    return _elements.size();
}

const matrix &dataset::get_features() const
{
    matrix features(4, 4, "dataset::features");

    // TODO return all the features concatenated.
    return features;
}

const matrix &dataset::get_labels() const
{
    matrix labels(4, 4, "dataset::labels");

    // TODO return all the labels concatenated.
    return labels;
}

#include <array>
std::pair<dataset, dataset> dataset::train_test_split(const float train_size_ratio /*= 0.8f*/)
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

    // Fill array with [0, "MULT_SIZE"] sequence, and shuffle it.
    std::array<size_t, dataset::MULT_SIZE> numbers;
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device generator;
    std::mt19937 distribution = std::mt19937(generator());
    std::shuffle(numbers.begin(), numbers.end(), distribution);
    // Select the "train_size" first indexes for the training set.
    for (size_t i = 0; i < size_; i ++)
    {
        if (i < train_size)
        {
            train.add(get(numbers[i]));
        }
        else
        {
            test.add(get(numbers[i]));
        }
    }

    return std::pair<dataset, dataset>(train, test);
}

dataset dataset::get_random_batch(const size_t batch_size)
{
    if (batch_size < 0 || size() < batch_size)
    {
        // Invalid.
        util::ERROR("dataset::get_random_batch", "Invalid @batch_size");
        util::ERROR_EXIT();
    }

    dataset batch;

    // Fill array with [0, "MULT_SIZE"] sequence, and shuffle it.
    std::array<size_t, dataset::MULT_SIZE> numbers;
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device generator;
    std::mt19937 distribution = std::mt19937(generator());
    std::shuffle(numbers.begin(), numbers.end(), distribution);
    // Select the "batch_size" first numbers as indexes.
    for (size_t i = 0; i < batch_size; i ++)
    {
        batch.add(get(numbers[i]));
    }

    return batch;
}

dataset *dataset::load_mult()
{
    auto data = new dataset();

    for (size_t i = 0; i < MULT_SIZE; i ++)
    {
        // TODO need to free pointers (element be pointer).
        auto features = new matrix(1, MULT_NB_FEATURES, 
                                   std::string("dataset::mult::features::") + std::to_string(i));
        auto labels = new matrix({ 1 }, 1, MULT_NB_LABELS, 
                                 std::string("dataset::mult::labels::") + std::to_string(i));

        for (size_t j = 0; j < MULT_NB_FEATURES; j ++)
        {
            features->get_data()[j] = (int) (std::rand() % MULT_MAX);
            labels->get_data()[0] *= (int) features->get_data()[j];
        }

        data->add(*features, *labels);
    }

    return data;
}

void dataset::print(dataset &d)
{
    size_t i = 1;

    for (auto &e: d.get_elements())
    {
        std::cout << ">>> n°" << (i ++) << " <<<" << std::endl; 
        element::print(e);
    }
}
