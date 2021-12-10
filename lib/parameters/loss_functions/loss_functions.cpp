//
// Created by Emilien Aufauvre on 20/11/2021.
//

#include "loss_functions.h"


matrix loss_functions::mean_square_error(matrix predictions, matrix labels)
{
    matrix error = matrix(predictions.get_dimensions());
    // TODO allocate matrix
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);

    loss_functions::__mean_square_error(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());

    return error;
}

matrix loss_functions::mean_absolute_error(matrix predictions, matrix labels)
{
    matrix error = matrix(predictions.get_dimensions());
    // TODO allocate matrix
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);

    loss_functions::__mean_absolute_error(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());

    return error;
}

matrix loss_functions::mean_bias_error(matrix predictions, matrix labels)
{
    matrix error = matrix(predictions.get_dimensions());
    // TODO allocate matrix
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);

    loss_functions::__mean_bias_error(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());

    return error;
}

matrix loss_functions::svm_loss(matrix predictions, matrix labels)
{
    matrix error = matrix(predictions.get_dimensions());
    // TODO allocate matrix
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);

    loss_functions::__svm_loss(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());

    return error;
}

matrix loss_functions::cross_entropy_loss(matrix predictions, matrix labels)
{
    matrix error = matrix(predictions.get_dimensions());
    // TODO allocate matrix
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);

    loss_functions::__cross_entropy_loss(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());

    return error;
}
