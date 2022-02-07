//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "dataset.h"


using namespace cudaNN;


dataset::dataset() = default;

dataset::dataset(std::vector<entry> &entries):
        _entries(entries)
{
}

dataset::~dataset()
{
    //util::DEBUG("dataset::~dataset", "---");
}

void dataset::add(const matrix &features, const matrix &labels)
{
    _entries.emplace_back(features, labels);
}

void dataset::add(const entry &e)
{
    _entries.push_back(e);
}

entry &dataset::get(const size_t i)
{
    return _entries[i];
}

std::vector<entry> &dataset::get_entries()
{
    return _entries;
}

size_t dataset::size() const
{
    return _entries.size();
}

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

    auto train = dataset();
    auto test = dataset();
    size_t train_size = (float) size_ * train_size_ratio;

    // Fill array with [0, "size_"] sequence, and shuffle it.
    auto numbers = std::vector<size_t>(size_);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device generator;
    auto distribution = std::mt19937(generator());
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

    return { train, test };
}

dataset dataset::get_random_batch(const size_t batch_size)
{
    if (batch_size < 0 || size() < batch_size)
    {
        // Invalid.
        util::ERROR("dataset::get_random_batch", "Invalid @batch_size");
        util::ERROR_EXIT();
    }

    auto batch = dataset();
    // Fill array with [0, "size()"] sequence, and shuffle it.
    auto numbers = std::vector<size_t>(size());
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device generator;
    auto distribution = std::mt19937(generator());
    std::shuffle(numbers.begin(), numbers.end(), distribution);
    // Select the "batch_size" first numbers as indexes.
    for (size_t i = 0; i < batch_size; i ++)
    {
        batch.add(_entries[numbers[i]]);
    }

    return batch;
}

dataset dataset::load_mult()
{
    util::INFO("dataset::load_mult", "loading the mult dataset");

    auto data = dataset();

    for (size_t i = 0; i < MULT_SIZE; i ++)
    {
        auto features = matrix(1, MULT_NB_FEATURES,
                               "dataset::mult::features::" + std::to_string(i));
        auto labels = matrix({1 }, 1, MULT_NB_LABELS,
                             "dataset::mult::labels::" + std::to_string(i));

        for (size_t j = 0; j < MULT_NB_FEATURES; j ++)
        {
            features.get_data()[j] = ((float) std::rand() / (float) RAND_MAX) * (float) MULT_MAX;
            labels.get_data()[0] *= features.get_data()[j];
        }

        data.add(features, labels);
    }

    return data;
}

void dataset::print(dataset &d)
{
    size_t i = 1;

    for (const auto &e: d.get_entries())
    {
        std::cout << ">>> n°" << (i ++) << " <<<" << std::endl; 
        entry::print(e);
    }
}