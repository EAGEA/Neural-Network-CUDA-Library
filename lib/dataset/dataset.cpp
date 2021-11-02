//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "dataset.h"


dataset::dataset()
{
    // TODO
    _elements = new();
}

dataset::dataset(std::vec<element> elements)
{
    // TODO
    _elements = elements;
}

void dataset::add(matrix features, matrix labels)
{
    add(new element(features, labels));
}

void dataset::add(element elem)
{
    _elements.push_back(elem);
}

void dataset::remove(element elem)
{

}

void dataset::remove(size_t i)
{

}

std::vec<elements> dataset::get_elements()
{
    return _elements;
}

const size_t dataset::size()
{
    return std::min(_features.size_x(), _labels.size_x());
}

std::pair<dataset, dataset> dataset::train_test_split(float train_size = 0.8f)
{
    dataset train;
    dataset test;

    // TODO
    dataset test = new(_elements.copy());
    dataset train = new ();

    for (size_t i = 0; i < size() * train_size; i ++)
    {
        size_t random = getRandom() % test.size();
        train.add(test.at(random));
        test.remove(random);
    }

    return std::pair<dataset, dataset>(train, test);
}

dataset dataset::get_random_batch(size_t batch_size)
{
    // TODO
    dataset batch;
    std::vec<elements> elements = _elements.copy();

    for (size_t i = 0; i < batch_size; i ++)
    {
        size_t random = RANDOM() % elements.size();
        batch.add(elements.at(random));
        elements.clear(random);
    }

    return batch;
}

static dataset dataset::loadMNIST()
{
    dataset data;
    uint32_t num_items;
    uint32_t rows;
    uint32_t cols;
    char label;
    char *pixels = new char[rows * cols];

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

    return data;
}