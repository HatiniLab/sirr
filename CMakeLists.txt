Project(SIRR)
cmake_minimum_required(VERSION 3.5)
SET(CMAKE_BUILD_TYPE Release)
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})
add_definitions(" -fpermissive -fPIC  -std=c++11 -O3 ")

include_directories(./include/)
add_executable(sirr src/sirr.cpp src/TrackingAndDeconvolutionProject.cpp)

target_link_libraries(sirr ${ITK_LIBRARIES})
