//
// Created by Emilien Aufauvre on 31/10/2021.
//

#ifndef CUDANN_ELEMENT_H
#define CUDANN_ELEMENT_H


class element
{
    public:

        element(matrix _features, matrix _labels);

        const matrix get_features() const;
        const matrix get_labels() const;

    private:

        const matrix _features;
        const matrix _labels;
};


#endif //CUDANN_ELEMENT_H