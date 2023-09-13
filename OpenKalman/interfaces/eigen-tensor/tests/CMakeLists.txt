# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/


cmake_minimum_required(VERSION 3.10)

if (CXX_STANDARD GREATER_EQUAL 20)
  find_package(Eigen3 3.3.9 REQUIRED CONFIG)
else()
  find_package(Eigen3 3.3 REQUIRED CONFIG)
endif()

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(eigen3-tensor
        eigen-tensor-traits.test.cpp
        eigen-tensor-nary-operations.test.cpp
    )

target_link_libraries(eigen3-tensor ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(eigen3-tensor DISCOVERY_TIMEOUT 20)
