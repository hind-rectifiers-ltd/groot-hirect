cmake_minimum_required(VERSION 3.5.1)
project(demo-canfd VERSION 1.0.0 LANGUAGES C CXX)

add_executable(demo-canfd main.cpp)

AUX_SOURCE_DIRECTORY(./ DIR_SRCS)

#设置编译结果文件输出目录
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -w -fpermissive -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

message(STATUS "build type:")
message(STATUS "CXX_FLAGS = ${CMAKE_CXX_FLAGS} ")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} pthread)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} dl)
