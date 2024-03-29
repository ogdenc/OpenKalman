# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2020.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(coefficients
        coefficients.cpp
        coefficient-functions.cpp
    )

target_link_libraries(coefficients ${GTEST_BOTH_LIBRARIES} Threads::Threads)

gtest_discover_tests(coefficients DISCOVERY_TIMEOUT 10)
