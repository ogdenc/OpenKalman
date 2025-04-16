# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2020.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(collections
        tests.cpp
        stl-collections.cpp
        get.cpp
        tuple-utils.cpp
        views.cpp
        iota.cpp
        transform.cpp
    )

target_link_libraries(collections GTest::gtest_main Eigen3::Eigen Threads::Threads)

gtest_discover_tests(collections DISCOVERY_TIMEOUT 10)
