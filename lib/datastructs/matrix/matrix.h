//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_MATRIX_H
#define CUDANN_MATRIX_H

#include "lib/util/util.h" 

#include <cstddef>
#include <utility>


/**
 * Matrix representation, allocating memory on both host and
 * device, and implementing computations with CUDA.
 * A matrix of size N*M has N rows (first dimensions)
 * and M columns (second dimension).
 */
class matrix
{
    public:

        explicit matrix(std::pair<size_t, size_t> dimensions);
        matrix(size_t x, size_t y);

        /**
         * Allocate memory on both host and device.
         */
        void allocate();

        /**
         * Free the memory previously allocated on the host and device.
         */
        void free();

        matrix add(const matrix &m) const;
        matrix multiply(const matrix &m) const;

        const std::pair<size_t, size_t> get_dimensions() const;

        const float *get_host_data() const;
        const float *get_device_data() const;
        float *get_host_data();
        float *get_device_data();

        void set_host_data(const size_t i, float f);
        void set_host_data(const size_t i, const size_t j, float f);
        void set_device_data(const size_t i, float f);
        void set_device_data(const size_t i, const size_t j, float f);

        bool compare_host_data(const matrix &m) const;
        bool compare_device_data(const matrix &m) const;
        matrix& operator=(const matrix &m);

    private:

        const std::pair<size_t, size_t> _dimensions;

        float *_device_data;
        float *_host_data;

        bool _allocated;
};

/**
 * CUDA function wrappers.
 */

void __allocate(const std::pair<size_t, size_t> dimensions, float *&device_data);
void __free(float *&device_data);
void __add(const dim3 block_dims, const dim3 thread_dims,
           float *output, 
           const float *data1, const float *data2,
           const size_t nb_rows, const size_t nb_cols);
void __multiply(const dim3 block_dims, const dim3 thread_dims,
                float *output, 
                const float *data1, const float *data2,
                const size_t nb_rows_1, const size_t nb_cols_1,
                const size_t nb_rows_2, const size_t nb_cols_2);

/**
 * Boolean operators are working only on host memory.
 * Aggregation operators are working only on device memory.
 */

bool operator==(const matrix &m1, const matrix &m2);
bool operator!=(const matrix &m1, const matrix &m2);
matrix operator+(const matrix &m1, const matrix& m2);
matrix operator*(const matrix &m1, const matrix& m2);


#endif //CUDANN_MATRIX_H
