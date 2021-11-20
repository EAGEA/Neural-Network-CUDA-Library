//
// Created by Emilien Aufauvre on 02/11/2021.
//


layer::layer(std::pair<size_t, size_t> dimensions)
{
    _dimensions = dimensions;
}

const std::pair<size_t, size_t> get_dimensions() const
{
    return _dimensions;
}