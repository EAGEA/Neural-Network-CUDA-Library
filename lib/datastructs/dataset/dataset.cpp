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

void dataset::add(matrix features, matrix labels)
{
    add(element(features, labels));
}

void dataset::add(element elem)
{
    _elements.push_back(elem);
}

void dataset::remove(element elem)
{
    // TODO
    /**
    _elements.erase(std::remove(_elements.begin(), _elements.end(), elem), 
            _elements.end());
            */
}

void dataset::remove(size_t i)
{
    //TODO
    /*
    _elements.erase(_elements.begin() + i);
    */
}

element dataset::get(size_t i)
{
    return _elements.at(i);
}

std::vector<element> dataset::get_elements()
{
    return _elements;
}

size_t dataset::size() const
{
    return _elements.size();
}

matrix get_features()
{
    matrix features(4, 4);

    // TODO return all the features concatenated.
    return features;
}

std::pair<dataset, dataset> dataset::train_test_split(float train_size_ratio /*= 0.8f*/)
{
    size_t size_ = size();
    size_t train_size = size_ * train_size_ratio;

    if (train_size < 0 || 1 < train_size)
    {
        // Invalid.
        util::ERROR("dataset::train_test_split", "Invalid @training_size_ratio");
        util::ERROR_EXIT();
    }

    dataset train;
    dataset test;
    size_t nb_selected = 0;

    for (size_t i = 0; (i < size_) && (nb_selected < train_size); i ++)
    {
        float probability_to_be_selected = (train_size - nb_selected) / (size_ - i);
        float random = ((float) std::rand() / (float) (RAND_MAX)) * (float) (size_ - i);

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

    for (size_t i = 0; (i < size_) && (nb_selected < batch_size); i ++)
    {
        float probability_to_be_selected = (batch_size - nb_selected) / (size_ - i);
        float random = ((float) std::rand() / (float) (RAND_MAX)) * (float) (size_ - i);

        if (random <= probability_to_be_selected)
        {
            batch.add(get(i));
            nb_selected ++;
        }
    }

    return batch;
}

dataset dataset::loadMNIST()
{
    dataset data;
    // TODO
    /*
    uint32_t num_items;
    uint32_t rows;
    uint32_t cols;

    // Open files.
    std::ifstream image_file(MNIST_IMAGE_FILENAME, std::ios::in | std::ios::binary);
    std::ifstream label_file(MNIST_LABEL_FILENAME, std::ios::in | std::ios::binary);
    // Read infos.
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    num_items = util::swap_endian(num_items);
    rows = util::swap_endian(rows);
    cols = util::swap_endian(cols);

    char label;
    char *pixels = new char[rows * cols];
    // Read pixels and label of each image.
    for (int i = 0; i < num_items; i ++)
    {
        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);
        // TODO /////////
        // convert it to cv Mat, and show it
        cv::Mat image_tmp(rows,cols,CV_8UC1,pixels);
        // resize bigger for showing
        cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
        cv::imshow(sLabel, image_tmp);
        cv::waitKey(0);
        /////////////////
    }

    delete[] pixels;
*/
    return data;
}
