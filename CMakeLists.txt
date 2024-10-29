# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

project(OpenKalman
        DESCRIPTION "An open-source, heading-only library for Kalman filters and other recursive filters"
        LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/OpenKalman)

find_package(Threads)

#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#  include_directories(SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

#find_package(OpenCV REQUIRED)
#include_directories(SYSTEM ${OPENCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(MyCXX_FLAGS "")

#if(${CMAKE_CXX_STANDARD} LESS 20)
#    set(MyCXX_FLAGS "${MyCXX_FLAGS}")
#endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MyCXX_FLAGS}")

add_subdirectory(OpenKalman)
add_subdirectory(docs)

