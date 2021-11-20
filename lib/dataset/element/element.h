//
// Created by Emilien Aufauvre on 31/10/2021.
//

#ifndef CUDANN_ELEMENT_H
#define CUDANN_ELEMENT_H


/**
 * Element in a dataset.
 */
class element
{
    public:

        element(matrix features, matrix labels);

        const matrix get_features() const;
        const matrix get_labels() const;

        bool compare_features(const matrix &features) const;
        bool compare_labels(const matrix &labels) const;
        bool compare(const element &e) const;

        bool operator==(const element &e1, const element &e2);
        bool operator!=(const element &e1, const element &e2);

    private:

        const matrix _features;
        const matrix _labels;
};


#endif //CUDANN_ELEMENT_H