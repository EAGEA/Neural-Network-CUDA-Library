//
// Created by Emilien Aufauvre on 20/11/2021.
//

#include "loss_functions.h"


using namespace cudaNN;


matrix loss_functions::mean_square_error(matrix predictions, matrix labels)
{
    auto error = matrix(predictions.get_dimensions(),
                        "loss_functions::mean_square_error");
    auto cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);
    /*
    // Prepare data of operands.
    predictions.copy_host_to_device();
    labels.copy_host_to_device();
    // Do the computations.
    loss_functions_cuda::mean_square_error(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());
    // Retrieve data of output.
    error.copy_device_to_host();
*/
    return error;
}

matrix loss_functions::mean_absolute_error(matrix predictions, matrix labels)
{
    auto error = matrix(predictions.get_dimensions(),
                        "loss_functions::mean_absolute_error");
    auto cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);
    /*
    // Prepare data of operands.
    predictions.copy_host_to_device();
    labels.copy_host_to_device();
    // Do the computations.
    loss_functions_cuda::mean_absolute_error(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());
    // Retrieve data of output.
    error.copy_device_to_host();
     */

    return error;
}

matrix loss_functions::mean_bias_error(matrix predictions, matrix labels)
{
    auto error = matrix(predictions.get_dimensions(),
                        "loss_functions::mean_bias_error");
    auto cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);
    /*
    // Prepare data of operands.
    predictions.copy_host_to_device();
    labels.copy_host_to_device();
    // Do the computations.
    loss_functions_cuda::mean_bias_error(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());
    // Retrieve data of output.
    error.copy_device_to_host();
     */

    return error;
}

matrix loss_functions::svm_loss(matrix predictions, matrix labels)
{
    auto error = matrix(predictions.get_dimensions(),
                        "loss_functions::svm_loss");
    auto cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);
    /*
    // Prepare data of operands.
    predictions.copy_host_to_device();
    labels.copy_host_to_device();
    // Do the computations.
    loss_functions_cuda::svm_loss(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());
    // Retrieve data of output.
    error.copy_device_to_host();
*/
    return error;
}

matrix loss_functions::cross_entropy_loss(matrix predictions, matrix labels)
{
    auto error = matrix(predictions.get_dimensions(),
                        "loss_functions::cross_entropy_loss");
    auto cuda_dims = util::get_cuda_dims(
            predictions.get_dimensions().first,
            predictions.get_dimensions().second);
    /*
    // Prepare data of operands.
    predictions.copy_host_to_device();
    labels.copy_host_to_device();
    // Do the computations.
    loss_functions_cuda::cross_entropy_loss(
            cuda_dims.first, cuda_dims.second,
            error.get_device_data(),
            predictions.get_device_data(),
            labels.get_device_data());
    // Retrieve data of output.
    error.copy_device_to_host();
*/
    return error;
}
