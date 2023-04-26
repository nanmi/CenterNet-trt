cmake_minimum_required(VERSION 3.10)
project(trt_infer)
set(CMAKE_CXX_STANDARD  11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -Wall -O2 -Wno-deprecated-declarations -Wno-unused-function")
set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda --expt-relaxed-constexpr -std=c++11 -O2")
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)

include_directories(${PROJECT_SOURCE_DIR})

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT-8.0.1.6)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)


include_directories(${PROJECT_SOURCE_DIR}/dcnv2_trt)
cuda_add_executable(ctdet ${PROJECT_SOURCE_DIR}/ctdet_infer.cpp
                      ${PROJECT_SOURCE_DIR}/dcnv2_trt/DCNv2Plugin.cpp
                      ${PROJECT_SOURCE_DIR}/dcnv2_trt/dcn_v2_im2col_cuda.cu
                      ${PROJECT_SOURCE_DIR}/decode.cu
                      )
                      
target_link_libraries(ctdet nvinfer)
target_link_libraries(ctdet cudart)
target_link_libraries(ctdet ${OpenCV_LIBS} nvparsers cublas nvinfer_plugin cudnn)
