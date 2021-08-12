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

add_executable(transformation
        transformations.gtest.cpp
        general.cpp
        linear.cpp
        stochastic.cpp
        linear-fd.cpp
    )

target_link_libraries(transformation ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(transformation)
