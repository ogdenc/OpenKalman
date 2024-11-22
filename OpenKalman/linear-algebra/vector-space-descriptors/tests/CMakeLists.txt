# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2024.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(descriptors
        static-descriptors.cpp
        static_concatenate.cpp
        static_reverse.cpp
        static_canonical_form.cpp
        equivalent_to.cpp

        #prefix_base_of.cpp

        #static-descriptor-manipulation.cpp
        #dynamic-descriptors.cpp
        #dynamic-descriptor-manipulation.cpp

        #functions.cpp

        #vector-space-descriptor-functions.cpp
    )

target_link_libraries(descriptors GTest::gtest_main Eigen3::Eigen Threads::Threads)

gtest_discover_tests(descriptors DISCOVERY_TIMEOUT 10)
