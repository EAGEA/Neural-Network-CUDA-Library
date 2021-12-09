//
// Created by Emilien Aufauvre on 31/10/2021.
//

#include "element.h"


element::element(const matrix &features, const matrix &labels): 
    _features(features), 
    _labels(labels)
{
}

const matrix element::get_features() const
{
    return _features;
}

const matrix element::get_labels() const
{
    return _labels;
}

bool element::compare_features(const matrix &features) const
{
    return _features.compare_host_data(features);
}

bool element::compare_labels(const matrix &labels) const
{
    return _labels.compare_host_data(labels);
}

bool element::compare(const element &e) const
{
    return compare_features(e.get_features()) && compare_labels(e.get_labels());
}

element &element::operator=(element &e)
{
    if (this == &e)
    {
        return *this;
    }

    // Re-assign for const members.
    this->~element();
    new (this) element(e.get_features(), e.get_labels());

    return *this;
}

bool operator==(const element &e1, const element &e2)
{
    return e1.compare(e2);
}

bool operator!=(const element &e1, const element &e2)
{
    return ! e1.compare(e2);
}
