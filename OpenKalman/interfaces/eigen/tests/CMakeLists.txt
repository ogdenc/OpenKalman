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

add_executable(eigen3
        eigen-traits.test.cpp
        eigen-nary-op-traits.test.cpp
        eigen-vector-wise.test.cpp
        eigen-matrix.test.cpp
        eigen-diagonal.test.cpp
        eigen-nary-operations.test.cpp
        eigen-reductions.test.cpp
        eigen-blocks.test.cpp
        eigen-chipwise-operations.test.cpp
        eigen-basic-linear-algebra.test.cpp
        eigen-rank-update.test.cpp
        eigen-solve.test.cpp
        eigen-decomposition.test.cpp
    )

target_link_libraries(eigen3 ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(eigen3 DISCOVERY_TIMEOUT 20)
