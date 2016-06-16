Project(SIRR)
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET(CMAKE_BUILD_TYPE Release)
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})
add_definitions(" -fpermissive -fPIC  -std=c++11 -O3 ")

include_directories(./include/)
add_executable(sirr src/sirr.cpp src/TrackingAndDeconvolutionProject.cpp)

target_link_libraries(sirr ${ITK_LIBRARIES})

add_executable(splitinput src/splitinput.cpp src/TrackingAndDeconvolutionProject.cpp)

target_link_libraries(splitinput ${ITK_LIBRARIES})
