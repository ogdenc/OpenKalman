# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED CONFIG)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(linear
        linear.gtest.cpp
    linear.2x2.cpp
    linear.2x3.cpp
    sample.2x2.cpp
    sample.2x3.cpp
    sample.3x2.cpp
    sample.4x3.cpp
    identity.cpp
    MCT.cpp
    )

target_link_libraries(linear ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(linear)
