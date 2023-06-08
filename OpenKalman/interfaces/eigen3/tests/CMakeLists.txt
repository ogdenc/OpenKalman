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

add_executable(eigen3
    eigen3-traits.test.cpp
    eigen3-nary-op-traits.test.cpp
    eigen3-matrix.test.cpp
    eigen3-diagonal.test.cpp
    eigen3-elementwise-operations.test.cpp
    eigen3-reductions.test.cpp
    eigen3-blocks.test.cpp
    eigen3-chipwise-operations.test.cpp
    eigen3-basic-linear-algebra.test.cpp
    eigen3-rank-update.test.cpp
    eigen3-solve.test.cpp
    eigen3-decomposition.test.cpp
    )

target_link_libraries(eigen3 ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen Threads::Threads)

gtest_discover_tests(eigen3 DISCOVERY_TIMEOUT 20)
