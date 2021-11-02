//
// Created by Emilien Aufauvre on 02/11/2021.
//


layer::layer(std::pair<size_t, sizet> dimensions)
{
    _dimensions = dimensions;
    _biases = new matrix(dimensions);
    _weights = new matrix(dimensions);

    _init_biases();
    _init_weights();
}

const std::pair<size_t, size_t> get_dimensions()
{
    return _dimensions;
}

void layer::_init_biases()
{
    for (int x = 0; x < _dimensions.first; x ++)
    {
        for (int y = 0; y < _dimensions.second; y ++)
        {
            _biases[y * _dimensions.first + x] = 0.f;
        }
    }
}

void layer::_init_weights()
{
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.f, 1.f);

    for (int x = 0; x < _dimensions.first; x ++)
    {
        for (int y = 0; y < _dimensions.second; y ++)
        {
            _weights[y * _dimensions.first + x] = normal_distribution(generator);
        }
    }
}