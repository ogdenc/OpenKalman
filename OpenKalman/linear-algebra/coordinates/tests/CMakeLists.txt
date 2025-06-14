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

add_executable(coordinates
        integral-types.cpp
        Dimensions.cpp
        #Distance.cpp
        #Angle.cpp
        #Inclination.cpp
        #Polar.cpp
        #Spherical.cpp
        #fixed_pattern.cpp
        #dynamic_pattern.cpp
        #statistical-transformations-fixed.cpp
        #statistical-transformations-dynamic.cpp
        #compare_descriptors.cpp
        #compare_patterns.cpp
        #comparison.cpp
        #arithmetic.cpp
        #slice.cpp

        #static-descriptor-manipulation.cpp
        #dynamic-descriptor-manipulation.cpp


    )

target_link_libraries(coordinates GTest::gtest_main Eigen3::Eigen Threads::Threads)

gtest_discover_tests(coordinates DISCOVERY_TIMEOUT 10)
