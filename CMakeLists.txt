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
        "lib/datastructs/dataset/dataset.cpp"
        "lib/datastructs/dataset/element/element.cpp"
        "lib/datastructs/matrix/matrix.cpp"
        "lib/datastructs/matrix/matrix.cu"
        "lib/models/neural_network/neural_network.cpp"
        "lib/models/neural_network/layers/layer.cpp"
        "lib/models/neural_network/layers/linear_layer.cpp"
        "lib/models/neural_network/layers/linear_layer.cu"
        "lib/parameters/activation_functions/activation_functions.cpp"
        "lib/parameters/activation_functions/activation_functions.cu"
        "lib/parameters/loss_functions/loss_functions.cpp"
        "lib/parameters/loss_functions/loss_functions.cu"
        "lib/util/util.cpp"
        )
    # Build tests ######################################################
    add_executable(nn_test tests/nn.cpp)
    target_link_libraries(nn_test CudaNN) 
    ###
    add_executable(matrix_test tests/matrix.cpp)
    target_link_libraries(matrix_test CudaNN) 
endif ()
########################################################################
########################################################################
