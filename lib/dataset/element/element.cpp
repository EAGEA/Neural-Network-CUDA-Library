//
// Created by Emilien Aufauvre on 31/10/2021.
//

#include "element.h"


element::element(matrix features, matrix labels)
{
    _features = features;
    _labels = labels;
}

const matrix element::get_features()
{
    return _features;
}

const matrix element::get_labels()
{
    return _labels;
}