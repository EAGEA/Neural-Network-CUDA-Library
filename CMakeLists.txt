cmake_minimum_required(VERSION 3.16)

# Set the project name and version ######################
project(CudaNN LANGUAGES CXX)
#########################################################
#########################################################
# Set standards #########################################
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#########################################################
#########################################################
# Include the whole project #############################
include_directories(${PROJECT_SOURCE_DIR})
#########################################################
#########################################################
# CUDA Libraries ########################################
find_package(CUDA)


set(MY_SOURCES
        "lib/datastructs/dataset/dataset.cpp"
        "lib/datastructs/dataset/element/element.cpp"
        "lib/datastructs/matrix/matrix.cpp"
        "lib/datastructs/matrix/matrix.cu"
        "lib/models/neural_network/neural_network.cpp"
        "lib/models/neural_network/layers/layer.cpp"
        "lib/models/neural_network/layers/linear_layer.cpp"
        "lib/models/neural_network/layers/linear_layer.cu"
        "lib/parameters/activation_functions/activation_functions.cpp"
        "lib/parameters/loss_functions/loss_functions.cpp"
        "lib/parameters/loss_functions/loss_functions.cu"
        "lib/util/util.cpp"
        "tests/main.cpp"
    )

if (CUDA_FOUND)
    enable_language(CUDA)
    #    include_directories(${CUDA_INCLUDE_DIRS} )
    add_executable(CudaNN1 ${MY_SOURCES}) 
endif ()
#########################################################
