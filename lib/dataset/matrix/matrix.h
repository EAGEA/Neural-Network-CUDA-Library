//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_MATRIX_H
#define CUDANN_MATRIX_H


class matrix
{
    public:

        matrix(std::pair<size_t, size_t> dimensions);
        matrix(size_t x, size_t y);

        const std::pair<size_t, size_t> get_dimensions() const;

        const float operator[](const size_t index);
        float& operator[](const size_t index);

    private:

        const std::pair<size_t, size_t> _dimensions;
};


#endif //CUDANN_MATRIX_H