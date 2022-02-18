//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_MATRIX_H
#define CUDANN_MATRIX_H

#include "lib/global.h"
#include "lib/util/util.h"

#include <vector_types.h> // To keep .cpp/.h extensions (cuda types).
#include <cstddef>
#include <string>
#include <utility>
#include <algorithm>
#include <initializer_list>

#define DEFAULT_ID "NaN"


namespace cudaNN
{
    /**
     * Matrix representation. Depending on the current configuration
     * (global.h) either do computations on the host or the device.
     * A matrix of size N*M has N rows and M columns (row major).
     */
    class matrix
    {
        public:

            matrix() = default;
            matrix(const matrix &m);
            matrix(const matrix &m, std::string id);
            matrix(size_t x, size_t y);
            matrix(size_t x, size_t y, std::string id);
            explicit matrix(std::pair<size_t, size_t> dimensions);
            matrix(std::pair<size_t, size_t> dimensions, std::string id);
            matrix(std::initializer_list<float> values, size_t x, size_t y);
            matrix(std::initializer_list<float> values, size_t x, size_t y, std::string id);
            matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions);
            matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions, std::string id);
            matrix(const float *values, std::pair<size_t, size_t> dimensions);
            matrix(const float *values, std::pair<size_t, size_t> dimensions, std::string id);
            ~matrix();

            void set_id(const std::string &id);

            const std::string &get_id() const;
            float *get_data() const;
            float *get_data();

            /**
             * @return - the number of rows, and columns of the matrix.
             */
            const std::pair<size_t, size_t> &get_dimensions() const;

            /**
             * @return - the number of values in the matrix.
             */
            size_t get_length() const;

            /**
             * @operators
             */
            matrix &operator=(const matrix &m);
            matrix &operator+=(const matrix &m);
            matrix operator+(const matrix &m);
            matrix &operator-=(const matrix &m);
            matrix operator-(const matrix &m);
            matrix &operator*=(const matrix &m);
            matrix operator*(const matrix &m);
            matrix &operator*=(float f);
            matrix operator*(float f);
            float &operator[](const int &i);
            const float &operator[](const int &i) const;
            bool operator==(const matrix &m) const;
            bool operator!=(const matrix &m) const;

            /**
             * @param v - a vector of the same dimensions as the current matrix
             * @return - the Hadamard product between the current matrix and "v".
             */
            matrix hadamard_product(const matrix &v);

            /**
             * @return - the sum of all the values in "_data".
             */
            float sum() const;

            /**
             * @return - the transpose of the matrix.
             */
            matrix transpose() const;

            /**
             * Print the given matrix (host memory).
             * @param m - the matrix concerned.
             */
            static void print(const matrix &m);

        private:

            void _allocate(const std::pair<size_t, size_t> &dimensions);
            void _free();

            std::string _id;
            std::pair<size_t, size_t> _dimensions;
            float *_data = nullptr;
    };


    /**
     * Cuda functions to be executed on device.
     */
    namespace matrix_parallel
    {
        void start_operation(const matrix &m, float **device_data);
        void end_operation(const matrix &m, float **device_data);
        void add(const matrix &m1, const matrix &m2);
        void subtract(const matrix &m1, const matrix &m2);
        void multiply(const matrix &m,
                      const matrix &m1, const matrix &m2);
        void multiply(const matrix &m, float f);
        void do_hadamard_product(const matrix &v1, const matrix &v2);
        void do_sum(float *result, const matrix &m);
        void do_transpose(matrix &result, const matrix &m);
    }


    /**
     * C++ functions to be executed on host.
     */
    namespace matrix_sequential
    {
        void add(const matrix &m1, const matrix &m2);
        void subtract(const matrix &m1, const matrix &m2);
        void multiply(const matrix &m,
                      const matrix &m1, const matrix &m2);
        void multiply(const matrix &m, float f);
        void do_hadamard_product(const matrix &v1, const matrix &v2);
        void do_sum(float *result, const matrix &m);
        void do_transpose(matrix &result, const matrix &m);
    }
}


#endif //CUDANN_MATRIX_H