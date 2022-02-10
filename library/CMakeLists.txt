cmake_minimum_required(VERSION 3.16)

# Set the project name and version #####################################
project(CudaNN LANGUAGES CXX)
# Set standards ########################################################
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Set base repository ##################################################
include_directories(${PROJECT_SOURCE_DIR})
# Include CUDA Libraries ###############################################
find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    # Build Library ####################################################
    add_library(CudaNN
            "lib/data_structures/dataset/dataset.cpp"
            "lib/data_structures/dataset/entry/entry.cpp"
            "lib/data_structures/matrix/matrix.cpp"
            "lib/data_structures/matrix/matrix_parallel.cu"
            "lib/data_structures/matrix/matrix_sequential.cpp"
            "lib/models/neural_network/neural_network.cpp"
            "lib/models/neural_network/layers/layer.cpp"
            "lib/functions/function.cpp"
            "lib/functions/activation_functions/activation_functions_parallel.cu"
            "lib/functions/activation_functions/activation_functions_sequential.cpp"
            "lib/functions/loss_functions/loss_functions_parallel.cu"
            "lib/functions/loss_functions/loss_functions_sequential.cpp"
            "lib/util/util.cpp"
            tests/debug_network.cpp)
    # Build tests ######################################################
    add_executable(test_matrix tests/matrix.cpp)
    target_link_libraries(test_matrix CudaNN)
    ###
    add_executable(test_activation_functions tests/activation_functions.cpp)
    target_link_libraries(test_activation_functions CudaNN)
    ###
    add_executable(test_loss_functions tests/loss_functions.cpp)
    target_link_libraries(test_loss_functions CudaNN)
    ###
    add_executable(test_neural_network tests/neural_network.cpp)
    target_link_libraries(test_neural_network CudaNN)
    ###
    add_executable(test_debug_network tests/debug_network.cpp)
    target_link_libraries(test_debug_network CudaNN)
    ###
    add_executable(test_op_time tests/op_time.cpp)
    target_link_libraries(test_op_time CudaNN)
    ###
endif ()