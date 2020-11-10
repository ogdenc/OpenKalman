# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.7)
project(OpenKalman)
include_directories(${PROJECT_SOURCE_DIR}/OpenKalman)

find_package(Threads REQUIRED)

#find_package(OpenCV REQUIRED)
#include_directories(SYSTEM ${OPENCV_INCLUDE_DIRS})

#enable_testing()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MyCXX_FLAGS "-g -O2 -Wall -ftemplate-backtrace-limit=0")

if(${CMAKE_CXX_STANDARD} LESS 20)
    set(MyCXX_FLAGS "${MyCXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MyCXX_FLAGS}")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall -pthread -fopenmp -ftemplate-backtrace-limit=0")

enable_testing()

add_subdirectory(OpenKalman)
add_subdirectory(tests)
