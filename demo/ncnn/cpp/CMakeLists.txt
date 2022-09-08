cmake_minimum_required(VERSION 3.8)
project(yolox-ncnn)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(yolox-pnnx
        yolox-pnnx.cpp
        )

target_link_libraries(yolox-pnnx
        ${OpenCV_LIBS}
        ncnn
        )

add_executable(yolox-ncnn
        yolox.cpp
        )

target_link_libraries(yolox-ncnn
        ${OpenCV_LIBS}
        ncnn
        )
