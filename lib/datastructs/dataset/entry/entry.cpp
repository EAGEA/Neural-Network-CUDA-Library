//
// Created by Emilien Aufauvre on 31/10/2021.
//

#include "entry.h"


using namespace cudaNN;
using namespace matrix_operators;


entry::entry(const matrix *features, const matrix *labels):
        _features(features),
        _labels(labels)
{
}

entry::~entry()
{
    util::DEBUG("entry::entry "---");
    delete _features;
    delete _labels;
}

const matrix &entry::get_features() const
{
    return *_features;
}

const matrix &entry::get_labels() const
{
    return *_labels;
}

bool entry::compare_features(const matrix &features) const
{
    return *_features == features;
}

bool entry::compare_labels(const matrix &labels) const
{
    return *_labels == labels;
}

bool entry::compare(const entry &e) const
{
    return compare_features(e.get_features()) && compare_labels(e.get_labels());
}

void entry::print(const entry &e)
{
    std::cout << "> features <" << std::endl;
    matrix::print(e.get_features());
    std::cout << "> labels <" << std::endl;
    matrix::print(e.get_labels());
}

bool operator==(const entry &e1, const entry &e2)
{
    return e1.compare(e2);
}

bool operator!=(const entry &e1, const entry &e2)
{
    return ! e1.compare(e2);
}