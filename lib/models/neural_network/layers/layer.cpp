//
// Created by Emilien Aufauvre on 02/11/2021.
//

#include "layer.h"
#include "lib/datastructs/matrix/matrix.h"

#include <cstddef>


layer::layer(const size_t nb_neurons): 
    _size(nb_neurons)
{
}

const size_t layer::size() const
{
    return _size;
}
