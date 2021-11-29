//
// Created by Emilien Aufauvre on 02/11/2021.
//


layer::layer(size_t nb_neurons)
: _size(nb_neurons)
{
}

const size_t size() const
{
    return _size;
}