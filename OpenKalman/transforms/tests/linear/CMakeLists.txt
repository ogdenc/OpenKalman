# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(linear
    linear.gtest.cpp
    MCT.cpp
    identity.cpp
    sample.4x3.cpp
    sample.3x2.cpp
    sample.2x3.cpp
    sample.2x2.cpp
    linear.2x3.cpp
    linear.2x2.cpp
    )

target_link_libraries(linear GTest::gtest_main Eigen3::Eigen Threads::Threads)

gtest_discover_tests(linear DISCOVERY_TIMEOUT 10)
