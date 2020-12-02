# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.9)

project(OpenKalman
        DESCRIPTION "An open-source, heading-only library for Kalman filters and other recursive filters"
        LANGUAGES CXX)

include_directories(${PROJECT_SOURCE_DIR}/OpenKalman)

#find_package(Threads REQUIRED)

find_package(OpenMP)
if(OpenMP_FOUND)
  include_directories(SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)

#find_package(OpenCV REQUIRED)
#include_directories(SYSTEM ${OPENCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MyCXX_FLAGS "-g -O2 -Wall -ftemplate-backtrace-limit=0")

#if(${CMAKE_CXX_STANDARD} LESS 20)
#    set(MyCXX_FLAGS "${MyCXX_FLAGS}")
#endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MyCXX_FLAGS}")

enable_testing()

add_subdirectory(OpenKalman)
add_subdirectory(tests)
add_subdirectory(docs)
