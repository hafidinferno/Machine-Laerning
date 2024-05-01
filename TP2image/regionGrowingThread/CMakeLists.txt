cmake_minimum_required(VERSION 3.10)

project(Tp1)

set(OpenCV_DIR C:/Users/ACER/Desktop/TPIMG2/opencv/build)

find_package(OpenCV REQUIRED)

add_executable(Tp1 regionGrowingThread.cpp)

target_link_libraries(Tp1 ${OpenCV_LIBS})
