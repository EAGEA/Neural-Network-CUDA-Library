//
// Created by Emilien Aufauvre on 20/11/2021.
//

#include "loss_functions.h"


using namespace cudaNN;


float loss_functions::mean_squared_error(const matrix &predictions, const matrix &labels)
{
    if (predictions.get_dimensions() != labels.get_dimensions())
    {
        // Invalid.
        util::ERROR("loss_functions::mean_squared_error",
                    "predictions dimensions does not have same dimensions as labels");
        util::ERROR_EXIT();
    }

    auto errors = matrix(predictions.get_dimensions(),
                         "loss_functions::mean_square_error::errors");
    auto cuda_dims = util::get_cuda_dims(predictions.get_dimensions());
    // Do the computations on device.
    loss_functions_cuda::mean_squared_error(
            cuda_dims.first, cuda_dims.second,
            errors,
            predictions, labels);

    return errors.sum() / (float) errors.get_length();
}

float loss_functions::mean_absolute_error(const matrix &predictions, const matrix &labels)
{
    if (predictions.get_dimensions() != labels.get_dimensions())
    {
        // Invalid.
        util::ERROR("loss_functions::mean_absolute_error",
                    "predictions dimensions does not have same dimensions as labels");
        util::ERROR_EXIT();
    }

    auto errors = matrix(predictions.get_dimensions(),
                         "loss_functions::mean_absolute_error::errors");
    auto cuda_dims = util::get_cuda_dims(predictions.get_dimensions());
    // Do the computations on device.
    loss_functions_cuda::mean_absolute_error(
            cuda_dims.first, cuda_dims.second,
            errors,
            predictions, labels);

    return errors.sum() / (float) errors.get_length();
}

float loss_functions::mean_bias_error(const matrix &predictions, const matrix &labels)
{
    if (predictions.get_dimensions() != labels.get_dimensions())
    {
        // Invalid.
        util::ERROR("loss_functions::mean_bias_error",
                    "predictions dimensions does not have same dimensions as labels");
        util::ERROR_EXIT();
    }

    auto errors = matrix(predictions.get_dimensions(),
                         "loss_functions::mean_bias_error::errors");
    auto cuda_dims = util::get_cuda_dims(predictions.get_dimensions());
    // Do the computations on device.
    // Do the computations on device.
    loss_functions_cuda::mean_bias_error(
            cuda_dims.first, cuda_dims.second,
            errors,
            predictions, labels);

    return errors.sum() / (float) errors.get_length();
}

float loss_functions::hinge_loss(const matrix &predictions, const matrix &labels)
{
    if (predictions.get_dimensions() != labels.get_dimensions())
    {
        // Invalid.
        util::ERROR("loss_functions::hinge_loss",
                    "predictions dimensions does not have same dimensions as labels");
        util::ERROR_EXIT();
    }

    auto errors = matrix(predictions.get_dimensions(),
                         "loss_functions::hinge_loss::errors");
    auto cuda_dims = util::get_cuda_dims(predictions.get_dimensions());
    // Do the computations on device.
    loss_functions_cuda::hinge_loss(
            cuda_dims.first, cuda_dims.second,
            errors,
            predictions, labels);

    return errors.sum();
}

float loss_functions::binary_cross_entropy_loss(const matrix &predictions, const matrix &labels)
{
    if (predictions.get_dimensions() != labels.get_dimensions())
    {
        // Invalid.
        util::ERROR("loss_functions::binary_cross_entropy_loss",
                    "predictions dimensions does not have same dimensions as labels");
        util::ERROR_EXIT();
    }

    auto errors = matrix(predictions.get_dimensions(),
                         "loss_functions::binary_cross_entropy_loss::errors");
    auto cuda_dims = util::get_cuda_dims(predictions.get_dimensions());
    // Do the computations on device.
    loss_functions_cuda::binary_cross_entropy_loss(
            cuda_dims.first, cuda_dims.second,
            errors,
            predictions, labels);

    return errors.sum();
}